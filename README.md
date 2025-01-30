# Real-Time-Anomaly-Segmentation for Road Scenes
This repository contains the code for the __Real-Time Anomaly Segmentation for Road Scenes__ project, developed as part of the Advanced Machine Learning course at the Polytechnic University of Turin. 


### Visualization of result
Visualization of the results produced by the ERFNet model with the MaxLogit inference method.


<p>
  <img src="samples/9_original.png" alt="Tractor" style="height:128px;width:256px;"/>
</p>
<p>
  <img src="samples/9_ground_truth_colored.png" alt="Tractor Ground Truth Anomaly" style="height:128px;width:256px;"/>
</p>
<p>
  <img src="samples/9_score.png" alt="Tractor Anomaly Scores" style="height:128px;width:256px;"/>
</p>
The top row contains the original images, the middle row represents the ground truth anomalies, and the bottom row visualizes the estimated anomaly scores, where hotter colors correspond to higher anomaly levels.


## Packages
For instructions, please refer to the README in each folder:

* [train](train) contains tools for training the network for semantic segmentation.
* [eval](eval) contains tools for evaluating/visualizing the network's output and performing anomaly segmentation.
* [imagenet](imagenet) Contains script and model for pretraining ERFNet's encoder in Imagenet.
* [trained_models](trained_models) Contains the trained models used in the papers. 
* [save](save) Contains the models trained for our experiments. 

## Requirements:

* [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0). 
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)
* **For testing the anomaly segmentation model**: Road Anomaly, Road Obstacle, and Fishyscapes dataset. All testing images are provided here [Link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Segmentantion models:
The repo provides the following pre-trained networks on the cityscapes dataset that can be used to perform anomaly segmentation on test anomaly datasets:
* __ERFNet__ 
* __BiSeNetV2__ 
* __Enet__ 
