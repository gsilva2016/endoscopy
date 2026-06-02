# Endoscopy Demo Powered by OpenVINO

This project demonstrates using OpenVINO to segment and detect polyps using open-source models pre-trained with the CVC-ColonDB dataset. The steps below will:

- Pre-train a YOLOv26n model which will have the ability to detect polyps
- The YOLO model will then be converted into an optimized OpenVINO FP16 version
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

Download the fine-tuned SAM 256x256model and optimize it with OpenVINO. 

```
git clone https://github.com/gsilva2016/medico-sam
cd medico-sam
```

```
wget https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download -O vit_b_medicosam.pt
```

or using Curl

```
curl -L -o vit_b_medicosam.pt https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download
```

```
python convert_model.py
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
python convert_model.py
cd ..
```

## Start the Demo

Execute the below command starting the demo for Yolo26n using OpenVINO accelerated GPU

```
python demo.py --device GPU --model_architecture yolo26n --image polyp.jpg
```

or loop the demo

```
python demo.py --device GPU --model_architecture yolo26n --loop --image polyp.jpg
```

# Acknowledgements

Special thanks to the following projects to make this demo possible.

| Project | Description |
| --- | --- |
| [Computational-cell-analytics](https://github.com/computational-cell-analytics/) | Fine-tuned  vit_b_medicosam.pt for SAM and medical imaging which segments well with smaller 256x256 image inputs |
| [SAMed](https://github.com/hitachinsk/samed/) | Modified SAM to load/work with smaller image sizes such as 256x256 |
| [YOLO-Colonoscopy](https://github.com/ArdeleanRichard/YOLO-Colonoscopy) | Easily train various YOLO architectures and versions with CVC-ColonDB dataset |
| [SAM OpenVINO Conversion Tutorial ](https://docs.openvino.ai/2024/notebooks/segment-anything-with-output.html) | Tutorial utilized to convert SAM to OpenVINO |
