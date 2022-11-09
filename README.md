# HA-RDet
Hybrid-Anchor Rotation Detector for Oriented Object Detection

![Alt text](model.png?raw=true "HA-RDet")

The repo is based on MMDetection and MMRotate

# Introduction
Oriented object detection in aerial images gains significant attention in computer vision and pattern recognition. Current state-of-the-art two-stage or one-stage methods commonly adopt Anchor-based strategies for region proposal generation using a redundant number of generated anchors for training, which is inefficient for the limited resources. At the same time, Anchor-free approaches are much faster but usually diminish a large number of training samples, excoriating the detection accuracy. In this paper, we present the Hybrid-Anchor Rotation Detector (HA-RDet), which aims to bridge the gap between Anchor-based and Anchor-free schemes for oriented object detection. In detail, we only use one preset anchor for each location in feature maps and implement further extra-oriented components that significantly boost the detection performance of HA-RDet and flexibly adapt to many well-designed oriented object detectors. Extensive experiments of HA-RDet and many other detectors are carried out on many well-known oriented datasets such as DOTA, DIOR-R and HRSC2016. Our HA-RDet achieves state-of-the-art results, competitively comparable with current Anchor-based methods, while the training and inference speed is asymptotically similar to Anchor-free competitors.
# Benchmark and Model Zoo
## DOTA dataset
Baseline HA-RDet
| Model        | Backbone             | MS  |Rotate|mAP    |configs|
|--------------|:--------------------:|:---:|:----:|:-----:|:-----:|
|HA-RDet       |ResNet50+FPN          |  -  |   -  |75.408 |       |
|HA-RDet       |ReResNet50+ReFPN      |  -  |   -  |75.676 |       |
|HA-RDet       |ResNext101_DCNv2+FPN  |  -  |   -  |77.012 |       |

High-quality detection HA-RDet
| Model                 | Backbone             | MS  |Rotate|mAP    |configs|
|-----------------------|:--------------------:|:---:|:----:|:-----:|:-----:|
|Oriented Cascade Head  |ResNet50+FPN          |  -  |   -  |46.64  |       |
|Oriented Dynamic Head  |ResNet50+FPN          |  -  |   -  |47.71  |       |
## DIOR-R dataset
| Model        | Backbone             | MS  |Rotate|mAP    |configs|
|--------------|:--------------------:|:---:|:----:|:-----:|:-----:|
|HA-RDet       |ResNext101_DCNv2+FPN  |  -  |   -  |65.3   |       |
## HRSC2016 dataset
| Model        | Backbone             | MS  |Rotate|mAP    |configs|
|--------------|:--------------------:|:---:|:----:|:-----:|:-----:|
|HA-RDet       |ResNext101_DCNv2+FPN  |  -  |   -  |90.2   |       |
# Installation

<summary> Data Tree </summary>

    HA-RDet
    ├── mmrotate
    ├── tools
    ├── configs
    ├── data
    │   ├── split_ss_dota
    │   │   ├── trainval
    │   │   ├── test
    │   ├── DIOR
    │   │   ├── trainval
    │   │   ├── test
    │   ├── HRSC
    │   │   ├── ImageSets
    │   │   ├── FullDataSets
Cloning repositories
```python
!git clone https://github.com/PhucNDA/HA-RDet.git
```
Install dependencies
```python
!pip install openmim
!mim install mmdet==2.25.0
!mim install mmrotate
```
## Training model

