# YOLO Series Object Detection Benchmark

## Introduction
Object Detection from Image (a.k.a. 2D Object Detection) is an important problem in Computer Vision. This page describes some of the popular Object Detection models.


## Datasets

**COCO** dataset: COCO is a large scale object detection, segmentation and captioning dataset. In this page we make use of the detection part of COCO dataset. COCO 2017 dataset is being used, unless otherwise mentioned. Accuracy evaluation is done on the validation split of COCO 2017, unless otherwise mentioned.


## Models 
The models are grouped in terms of repositories used to train them or the repositories through they are made available.

### YOLOv5 Models
- [Training Source Code & models]
- [Original source code](https://github.com/ultralytics/yolov5): We made changes on this repository to create the yolov5 fork.

|Dataset |Model Name                      |Input Size |GFLOPS    |Official FLOAT AP[0.5:0.95]%|FLOAT AP[0.5:0.95]%|Quant AP[0.5:0.95]%|
|--------|------------------------------- |-----------|----------|----------------------------|-------------------|-------------------|
|COCO    |YOLOv5n                         |640x640    |**4.6**   |   28.0                     | 27.0              | 26.2              |
|COCO    |YOLOv5l                         |640x640    |**109.6** |   49.0                     | 47.2              | 45.5              |
|COCO    |YOLOv5s6                        |1280x1280  |**17**  |   44.8                     | 43.6              | 42.0              |

### YOLOv4 Models
- [Training Source Code & models]
- [Original source code](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp): We made changes on this repository to create the yolov4 fork.

|Dataset |Model Name                      |Input Size |GFLOPS    |Official FLOAT AP[0.5:0.95]%|FLOAT AP[0.5:0.95]%|Quant AP[0.5:0.95]%|
|--------|------------------------------- |-----------|----------|----------------------------|-------------------|-------------------|
|COCO    |YOLOv4-csp                      |640x640    |**121**   |47.8                        |   47.0            | 46.3              |

If you need training source code and models, please contact AMD/Xlinx AI product marketing e-mail: mailto:amd_ai_mkt@amd.com