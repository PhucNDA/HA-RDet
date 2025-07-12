<div align="center">

  # Hybrid-Anchor Rotation Detector for Oriented Object Detection - ICCV'25 (SEA)
</div>

![image](((https://github.com/PhucNDA/HA-RDet/blob/main/images/hardet_architecture.png)))

## Introduction

Oriented object detection in aerial images poses a significant challenge due to their varying sizes and orientations. Current state-of-the-art detectors typically rely on either two-stage or one-stage approaches, often employing Anchor-based strategies, which can result in computationally expensive operations due to the redundant number of generated anchors during training. In contrast, Anchor-free mechanisms offer faster processing but suffer from a reduction in the number of training samples, potentially impacting detection accuracy. To address these limitations, we propose the Hybrid-Anchor Rotation Detector (HA-RDet), which combines the advantages of both anchor-based and anchor-free schemes for oriented object detection. By utilizing only one preset anchor for each location on the feature maps and refining these anchors with our Orientation-Aware Convolution technique, HA-RDet achieves competitive accuracies, including 75.41 mAP on DOTA-v1, 65.3 mAP on DIOR-R, and 90.2 mAP on HRSC2016, against current anchor-based state-of-the-art methods, while significantly reducing computational resources.

## Installation

Data preparation and download
* DOTA-v1.0: <a href="https://captain-whu.github.io/DOTA/dataset.html">download</a>
* HRSC2016: <a href="https://www.kaggle.com/datasets/guofeng/hrsc2016">download</a>
* DIOR-R: <a href="https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC">download</a>

```
HA-RDet
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_dota
│   │   ├── trainval
│   │   │    ├── annfiles
│   │   │    ├── images
│   │   ├── test
│   │   │    ├── annfiles
│   │   │    ├── images
│   ├── DIOR-R
│   │   ├── trainval
│   │   ├── test
│   ├── HRSC
│   │   ├── ImageSets
│   │   ├── FullDataSets
```

Our experiment relies on the <a href="https://github.com/open-mmlab/mmrotate">MMRotate</a> framework provided by <a href="https://github.com/open-mmlab">Open MMLab</a>.
MMRotate depends on <a href="https://pytorch.org/">PyTorch</a>, <a href="https://github.com/open-mmlab/mmcv">MMCV</a> and <a href="https://github.com/open-mmlab/mmdetection">MMDetection</a>. Quick steps for installation follows as:

* Git clone

```
git clone https://github.com/PhucNDA/HA-RDet
```

* Environment setup

```
conda create -n [NAME] python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate [NAME]
pip install openmim
mim install mmcv-full
mim install mmdet
cd 'Hybrid-Anchor-Rotation-Detector'
pip install -r requirements/build.txt
pip install -v -e .
```

## Training and Inference

* Training command:

```
python tools/train.py ${CONFIG_FILE} [optional arguments]

# Example:
python tools/train.py configs/ha_rdet/hardet_baseline_r50_fpn_1x_dota_le90.py
```

* Inference command for online submission:
```
python ./tools/test.py  \
    configs/ha_rdet/hardet_baseline_r50_fpn_1x_dota_le90.py \
    checkpoints/SOME_CHECKPOINT.pth --format-only \
    --eval-options submission_dir=[SAVE_FOLDER]
```

* Visualize the results
```
python ./tools/test.py  \
    configs/ha_rdet/hardet_baseline_r50_fpn_1x_dota_le90.py \
    checkpoints/SOME_CHECKPOINT.pth
    --show-dir [SAVE_FOLDER]
```

## Benchmark and Model Zoo

### DOTA-v1.0 dataset

| Model    |    Backbone       | #anchors              | VRAM (GB) | #params                   | FPS | mAP | Config | Download |
| ------ |:-------------:|:----------------------:|:-----------------------------------------------------:|:-------------------------:|:----:|:----:|:---:|:--:|
| S2A-Net| ResNet50+FPN | 1 | 4.6 | ~39M | 15.5 | 74.19 | - | - |
| Oriented R-CNN| ResNet50+FPN | 20 | 14.2 | ~41M | 13.5 | 75.69 | - | - |
| **HA-RDet (ours)** | ResNet50+FPN | 1 | 6.8 | ~56M | 12.1 | 75.41 | <a href="https://github.com/PhucNDA/HA-RDet/blob/main/configs/ha_rdet/hardet_baseline_r50_fpn_1x_dota_le90.py">config</a> | <a href="https://drive.google.com/file/d/1_8xUpm8dX5oypkBCiDuqYolG2u3_KuYW/view?usp=drive_link">model</a> / <a href="https://github.com/PhucNDA/HA-RDet/blob/main/logs/hardet_baseline_r50_fpn_1x_dota_le90.txt">log</a> |
| **HA-RDet (ours)** | ResNet101+FPN | 1 | - | - | - | 76.02 | <a href="https://github.com/PhucNDA/HA-RDet/blob/main/configs/ha_rdet/hardet_baseline_r101_fpn_1x_dota_le90.py">config</a> | <a href="https://drive.google.com/file/d/1Zm7eYrepwAmjJ0TaHti4d6Znn9T4bl__/view?usp=drive_link">model</a> / <a href="https://github.com/PhucNDA/HA-RDet/blob/main/logs/hardet_baseline_r101_fpn_1x_dota_le90.txt">log</a> |
| **HA-RDet (ours)** | ResNeXt101_DCNv2+FPN | 1 | - | - | - | 77.012 | <a href="https://github.com/PhucNDA/HA-RDet/blob/main/configs/ha_rdet/hardet_baseline_rx101_dcn_fpn_1x_dota_le90.py">config</a> | <a href="https://drive.google.com/file/d/1_29jCteJpW-13MxClbZP7eHuRY9HJPTH/view?usp=drive_link">model</a> / <a href="https://github.com/PhucNDA/HA-RDet/blob/main/logs/hardet_baseline_rx101_dcn_fpn_1x_dota_le90.txt">log</a> |

### HRSC2016

| Model | Backbone | #anchors | mAP (VOC 07) | mAP (VOC 12) |
|:-----:|:--------:|:-------:|:-------:|:-------:|
| S2A-Net | ResNet101+FPN | 1 | 90.17 | 95.01 |
| AOPG | ResNet101+FPN | 1 | 90.34 | 96.22 |
| **HA-RDet (ours)** | ResNeXt101_DCNv2+FPN | 1 | 90.2 | 95.32 |

### DIOR-R
| Model | Backbone | mAP |
|:-----:|:--------:|:---:|
| HA-RDet | ResNeXt101_DCNv2+FPN | 65.3 |

## Visualization
|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0014__1024__0___0.png)|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0182__1024__0___1109.png)|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0017__1024__0___1648.png)|
|-|-|-|
|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0031__1024__2472___0.png)|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0031__1024__2472___1648.png)|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0045__1024__0___0.png)|
|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0051__1024__3296___1648.png)|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0132__1024__0___0.png)|![image](https://github.com/PhucNDA/HA-RDet/blob/main/images/vis/P0145__1024__161___169.png)|
