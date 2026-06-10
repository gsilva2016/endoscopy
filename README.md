# Endoscopy Demo Powered by OpenVINO

This project demonstrates using OpenVINO to segment and detect polyps using open-source models pre-trained with the CVC-ColonDB dataset. The steps below will:

- Pre-train a YOLOv26n model which will have the ability to detect polyps
- The YOLO model will then be converted into an optimized OpenVINO FP16 version
- Download and optimize a U-Net 256x256 shaped model, via OpenVINO, which is fine-tuned for segmenting polyps
- Download and create a SAM based 256x256 shaped model which works well with segmenting polyps and after converted into an optimized OpenVINO version
- Finally, the last step will run the demo with detecting and segmenting a polyp from an image. This demo has an option to run with --loop so long-running performance can be viewed

## Create and activate isolated Python environment

```
conda create -y -n endo-ov python=3.12
```

```
conda activate endo-ov
```

```
git clone https://github.com/gsilva2016/endoscopy
cd endoscopy
pip install -r requirements.txt
```


## Create the OpenVINO Detection and Segmentation Models

Download the fine-tuned SAM 256x256model and optimize it with OpenVINO

```
git clone https://github.com/gsilva2016/medico-sam
cd medico-sam
```

Download with wget

```
wget https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download -O vit_b_medicosam.pt
```

or using curl

```
curl -L -o vit_b_medicosam.pt https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download
```

Convert the downloaded model to OpenVINO format

```
python convert_model.py
cd ..
```

Download the fine-tuned U-Net Residual ResNet34 model

```
git clone https://huggingface.co/RGarrido03/unet-residual-resnet34
del "unet-residual-resnet34\models\__init__.py"
python convert_unet_model.py
```

Download and install the YOLO-Colonoscopy for training YOLO with CVC-ColonDB dataset

```
git clone https://github.com/gsilva2016/YOLO-Colonoscopy.git
```

```
cd YOLO-Colonoscopy
```

```
pip install -r requirements.txt
```

Execute the below command for Yolo26n

```
python main_train.py --model_architecture yolo26n
cd ..
python convert_model.py --model_architecture yolo26n
```

## Start the Demo

Execute the below command starting the demo for Yolo26n + SAM using OpenVINO accelerated GPU

```
python demo.py --device GPU --model_architecture yolo26n --image polyp.jpg  --segmentation-model sam
```

or loop the demo

```
python demo.py --device GPU --model_architecture yolo26n --loop --image polyp.jpg  --segmentation-model sam
```
or use U-Net model instead of SAM for segmentation

```
python demo.py --device GPU --model_architecture yolo26n --loop --image polyp.jpg  --segmentation-model unet
```

or perform Yolo26n detection of polyps with no segmentation

```
python demo.py --device GPU --model_architecture yolo26n --loop --image polyp.jpg  --segmentation-model none
```

# Acknowledgements

Special thanks to the following projects to make this demo possible.

| Project | Description |
| --- | --- |
| [Computational-cell-analytics](https://github.com/computational-cell-analytics/) | Fine-tuned  vit_b_medicosam.pt for SAM and medical imaging which segments well with smaller 256x256 image inputs |
| [SAMed](https://github.com/hitachinsk/samed/) | Modified SAM to load/work with smaller image sizes such as 256x256 |
| [YOLO-Colonoscopy](https://github.com/ArdeleanRichard/YOLO-Colonoscopy) | Easily train various YOLO architectures and versions with CVC-ColonDB dataset |
| [SAM OpenVINO Conversion Tutorial ](https://docs.openvino.ai/2024/notebooks/segment-anything-with-output.html) | Tutorial utilized to convert SAM to OpenVINO |
| [U-Net Residual ResNet34 ](https://huggingface.co/RGarrido03/unet-residual-resnet34) | Very fast fine-tuned U-Net ResNet34 for 256x256 semantic segmentation of polyps  |
