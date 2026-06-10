import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'unet-residual-resnet34')))
import torch
import openvino as ov
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from models.backbones import ResNet34Backbone
from models.residual_unet import ResidualUNet

backbone = ResNet34Backbone(pretrained=False)
model = ResidualUNet(in_channels=3, num_classes=1, backbone=backbone)

weights_path = hf_hub_download(
    repo_id="RGarrido03/unet-residual-resnet34",
    filename="model.safetensors",
)
model.load_state_dict(load_file(weights_path), strict=False)
model.eval()
core = ov.Core()
ov_model = ov.convert_model(
 model,
 example_input=torch.zeros(1, 3, 256, 256),
 input=([1, 3, 256, 256],),
)
ov_model_path = "unet-residual-resnet.xml"
ov.save_model(ov_model, ov_model_path, compress_to_fp16=True)