import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'medico-sam')))

import argparse
import cv2
import time
import torch
import numpy as np
import signal
from pathlib import Path
from utils import *
from console import *

def app_wide_signal_handler(signum, frame):
    """Global handler to trap Ctrl+C across the application."""
    sys.exit(130) 


parser = argparse.ArgumentParser(description="Endo Detection and Segmentation Powered by OpenVINO")
parser.add_argument("--model_architecture", choices=["yolo11n", "yolo26n"], default="yolo26n")
parser.add_argument("--device", choices=["GPU", "CPU", "NPU"], default="GPU")
parser.add_argument("--disable-segmentation", action="store_true")
parser.add_argument("--image", default="polyp.jpg")
parser.add_argument("--loop", action="store_true", help="Perform inference forever (ctr+c to quit)")
args = parser.parse_args()

disable_segment = args.disable_segmentation
loop_like_video = args.loop
model_name = args.model_architecture[:-1]
model_path = f"./YOLO-Colonoscopy/runs/detect/results_data_CVC-ColonDB/runs_train/{model_name}/{args.model_architecture}/weights/best_openvino_model"
device = args.device

if device == "GPU":
    ultralytics_device = "intel:gpu"
elif device == "NPU":
    device = ultralytics_device="intel:npu"
else:
    device = ultralytics_device="intel:cpu"

try:
    from ultralytics import YOLO
except Exception as e:
    print(f"Please execute: pip install ultralytics")
    quit()

if not Path(model_path):
    print(f"Model not found: {model_path}")
    quit()

if not disable_segment:
    try:
        import openvino as ov
    except Exception as e:
        print(f"Please execute: pip install openvino")
        quit()

    sam_image_encoder_path = "sam_image_encoder.xml"
    sam_image_decoder_path = "sam_mask_predictor.xml"

    if not Path(sam_image_encoder_path):
        print(f"Model not found: {sam_image_encoder_path}")
        quit()
    if not Path(sam_image_decoder_path):
        print(f"Model not found: {sam_image_decoder_path}")
        quit()

    core = ov.Core()
    ov_path = Path(sam_image_encoder_path)    
    ov_encoder_model = core.read_model(ov_path)
    ov_encoder = core.compile_model(ov_encoder_model, device)
    ov_path = Path(sam_image_decoder_path)    
    ov_decoder_model = core.read_model(ov_path)
    ov_decoder = core.compile_model(ov_decoder_model, device)
    resizer = ResizeLongestSide(256)

# Load your PyTorch model
model = YOLO(model_path)

# Warm up....
print("Warm up inference...")
dummy_data = np.zeros((640, 640, 3), dtype=np.uint8) 
results = model.predict(source=dummy_data, imgsz=640, device=ultralytics_device)

if not disable_segment:
    dummy_data = (torch.randn(1, 3, 256, 256))
    ov_encoder(dummy_data)

    dummy_data = {
        "image_embeddings": torch.randn(1, 256, 16, 16, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=256, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    }
    ov_decoder(dummy_data)
print("Warm up finished....")

# Run inference
print(CBLUE + "Starting inference..." + CEND)
orig_image = cv2.imread(args.image)
image = cv2.resize(orig_image, (640, 640))

def perform_inference(image):
    start_t = time.time()
    results = model(image, device=ultralytics_device, verbose=False)
    print(f"Time taken for {CGREEN}detect (pre/post+inference){CEND}: {(time.time() - start_t):.3f} seconds")

    if not disable_segment:
        for result in results:
            image = results[0].plot() # orig_image
            start_seg_t = time.time()
            pp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preprocessed_image = preprocess_image(pp_image, resizer, 256)
            encoding_results = ov_encoder(preprocessed_image)
            image_embeddings = encoding_results[ov_encoder.output(0)]

            xyxy = result.boxes.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(x) for x in xyxy]

            input_box = np.array([x1,y1,x2,y2])
            input_point = np.array([[0, 0]])
            input_label = np.array([0])
            box_coords = input_box.reshape(2, 2)
            box_labels = np.array([2, 3])
            coord = np.concatenate([input_point, box_coords], axis=0)[None, :, :]
            label = np.concatenate([input_label, box_labels], axis=0)[None, :].astype(np.float32)
            coord = resizer.apply_coords(coord, image.shape[:2]).astype(np.float32)
            
            inputs = {
                "image_embeddings": image_embeddings,
                "point_coords": coord,
                "point_labels": label,
            }
            mask_results = ov_decoder(inputs)
            print(f"Time taken for {CGREEN}segmentation{CEND}: {(time.time() - start_seg_t):.3f} seconds")
            start_seg_pp_t = time.time()
            masks = mask_results[ov_decoder.output(0)]
            masks = postprocess_masks(masks, image.shape[:-1], resizer)
            masks = masks > 0.0
            print(f"Time taken for {CGREEN}segmentation post-processing{CEND}: {(time.time() - start_seg_pp_t):.3f} seconds")
            print(f"Total time taken for {CGREEN}detect and segmentation{CEND}: {(time.time() - start_t):.3f} seconds")
            return draw_mask(masks[0], image)
    else:
        return results[0].plot() # .show()

signal.signal(signal.SIGINT, app_wide_signal_handler)

while True:
    draw_img = perform_inference(image)
    cv2.imshow("OpenVINO Detect/Medico SAM Segmentation Polyp Demo", draw_img)
    
    if not loop_like_video:
        cv2.waitKey(0)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("You pressed 'q'. Exiting...")

cv2.destroyAllWindows()
print(CBLUE + "Demo Finished" + CEND)