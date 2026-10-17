# Vitis AI YOLOv5 Nano Deployment for KV260

This directory contains the actual Vitis AI YOLOv5 Nano model and application
materials used in the KV260 Autonomous Vehicle Vision project.

## Target Hardware

- AMD/Xilinx Kria KV260
- DPU: DPUCZDX8G_ISA1_B4096
- DPU frequency: 300 MHz
- PetaLinux 2022.2
- Vitis AI 3.0

## Model

The project uses a pretrained YOLOv5 Nano model supplied in a
Vitis-AI-compatible package for ZCU102, ZCU104 and KV260.

The deployed model is:

    model/yolov5_nano_pt.xmodel

Verified SHA256:

    2a45d4dabd309cb8ebb52493139bb4b871980228e4904705e0ce28517bc0b646

The model is not claimed to have been trained from scratch in this project.

## DPU Architecture

The DPU architecture description was taken directly from the DPUCZDX8G IP
instance used in the project's Vivado block design:

    dpu/arch.json

SHA256:

    6feb79ba36f1f030f7827d94cdd8b43ff03faec72dad8e4a4e4d72704ed8490e

## Application

### kv260-vision-app.cpp

Static-image application using VART/XIR for YOLO inference and the custom
HLS Sobel accelerator.

### kv260-vision-network-app.cpp

Receives JPEG frames over Ethernet and performs:

    Ethernet input
          |
          v
    ARM preprocessing
       /       \
      v         v
 Vitis AI DPU  HLS Sobel
 YOLOv5 Nano   accelerator
      \         /
       v       v
     ARM post-processing
          |
          v
 Detection + lane/edge visualization

### kv260-vision-video-app.cpp

Video/USB-camera version of the perception application.

## Important Architecture Detail

YOLO and Sobel are parallel processing branches.

The Sobel output is NOT passed into YOLO.

The DPU performs neural-network inference while the custom HLS Sobel
accelerator performs hardware edge extraction.

## Reference Sources

The application/reference directory contains Vitis AI YOLOv5 sample
applications used as reference material:

- test_accuracy_yolov5_mt.cpp
- test_jpeg_yolov5.cpp
- test_performance_yolov5.cpp
- test_video_yolov5.cpp

## Validated Performance

Representative measured results:

- HLS Sobel: approximately 1.74 ms/frame
- DPU inference: approximately 7.24 ms/frame
- Main compute capability: approximately 47 FPS
- Live CARLA end-to-end throughput: approximately 9.5 FPS

The live end-to-end rate is different from DPU-only performance because it
also includes frame transport, preprocessing, post-processing and other
pipeline overhead.

## Runtime Model Location on KV260

    /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel

## Integrity

Run:

    sha256sum -c SHA256SUMS.txt

to verify the files in this directory.
