# Brain-Tumor-Detection-using-Semantic-Segmentation

## Introduction
This repository is a semantic segementation implementation on scanned MRI images of the Brain.
This repository contains a small dataset on which the models are trained. The dataset is trained on various models of which the results are provided in the table down below.

## Features

| Flags  | Usage |
| ------------- | ------------- |
| ```--network``` | Define network (Default: custom)  | 
| ```--backbone```  | Define backbone	(Default: None)  |                                                                   
| ```--img_size```  | Define patch size (Default:256) |
| ```--weight_path```  | Set path to model weights  | 
| ```--data_path```  | Set path to data  | 
| ```--epoch```  | Set number of epochs (Default: 50)  |
| ```--verbose```  | Set verbose (Default: 1)  |
| ```--batch_size```  | Set Batch size (Default: 8)  |
| ```--validation_size```  | Set Validation size (Default: 0.1)  |
| ```-test_split```  | Set test size (Default: 0.2)  |
| ```--visualizer```  | Enable visualizer (Default: Not enabled)  |
| ```--score```  | Enable score calculation after training (Default: Not enabled)  |
| ```--test```  | Enable testing after training (Default: Not enabled)  |
| ```--augment```  | Enable Augmentation (Default: Not enabled) |

| Network  | BackBone |
| ------------- | ------------- |
| ```custom``` |```None``` |
| ```unet``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```, ```densenet121```, ```mobilenetv2``` |
| ```segnet``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```, ```densenet121```, ```mobilenetv2``` |
| ```deeplabv3``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```, ```densenet121```, ```mobilenetv2``` |

for ```pspnet``` image size must be divisible by 48, the image size will be adjusted accordingly.

## Installation
  ### Requirements
    -Python3
    -Cuda

  ### Install
    1. git clone https://github.com/dahalsweekar/Brain-Tumor-Detection-using-Semantic-Segmentation.git
    2. pip install -r requirements.txt 
    
## Training 

  > Training is set to early stopping
 ```
 python services/train.py --network unet --backbone vgg16 --img_size 256 --batch_size 8 --epoch 100 --score --data_path /content/drive/MyDrive/data/BTD_Dataset 
 ```
```
 python services/train_all.py 
 ```
## Models
  > Trained models are saved in ./models/

## Dataset

  > Root of the dataset, by default, is ./data/BTD_Dataset/
```
With augmentation: 

Train size: (174, 256, 256, 3)
Test size: (15, 256, 256, 3)

Without Augmentation: 

Train size: (58, 256, 256, 3)
Test size: (15, 256, 256, 3)
```
### Sample

Image             |  Label
:-------------------------:|:-------------------------:
![1](https://github.com/dahalsweekar/Deep-Weed-Segmentation/assets/99968233/43804f88-3f7d-4d67-85f1-60522f247f39)  |  ![1_label](https://github.com/dahalsweekar/Deep-Weed-Segmentation/assets/99968233/0662dd42-fb80-4b70-b836-5aa2b1304998)
```
|
|
|__./data/BTD_Dataset/
	|
	|___/images
		|
		|__*.jpg .png*
	|
	|___/labels
		|
		|__*.jpg .png*
```
## Results

| Backbone | Augmentation | Model | MeanIoU |Precision |Recall |F1-Score |Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|VGG16|False|Unet||||||
|||SegNet||||||
|||DeepLabV3+||||||
||True|Unet||||||
|||SegNet||||||
|||DeepLabV3+||||||
	

Accuracy             |  Loss
:-------------------------:|:-------------------------:
![accuracy](https://github.com/dahalsweekar/Deep-Weed-Segmentation/assets/99968233/a0d6b1f3-4938-4c80-9fd3-4ff72efa7d6c)  |  ![loss](https://github.com/dahalsweekar/Deep-Weed-Segmentation/assets/99968233/6b00143b-a5a0-41be-8101-5bfe67a06988)

### Prediction
![prediction](https://github.com/dahalsweekar/Deep-Weed-Segmentation/assets/99968233/d5833c9b-aed2-40fd-a8ff-07eb8b93ac58)

## Third-Party Implementations
 - Keras implementation
 - Segmentation model implementations: https://github.com/qubvel/segmentation_models
 - Advance Segmentation models: https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models
