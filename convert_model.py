import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Endo YOLO Model Creator")
parser.add_argument("--model_architecture", choices=["yolo11n", "yolo26n"], default="yolo26n")
args = parser.parse_args()

model_name = args.model_architecture[:-1]
model_path = f"./YOLO-Colonoscopy/runs/detect/results_data_CVC-ColonDB/runs_train/{model_name}/{args.model_architecture}/weights/best.pt"

try:
    from ultralytics import YOLO
except Exception as e:
    print(f"Please execute: pip install ultralytics")
    quit()

if not Path(model_path):
    print(f"File not found: {model_path}")
    quit()

# Load your PyTorch model
model = YOLO(model_path)

# Export to OpenVINO format
print("Optimizing detection model with OpenVINO")
model.export(format="openvino", half=True) 
print("Optimization completed")

# Output: 
# YOLO-Colonoscopy\runs\detect\results_data_CVC-ColonDB\runs_train\yolo26\yolo26s\weights\best_openvino_model