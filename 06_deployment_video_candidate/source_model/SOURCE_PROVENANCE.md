# YOLOv5 Nano Model Provenance

## Model

- Architecture: YOLOv5 Nano
- Framework: PyTorch
- Dataset: COCO 2017
- Input resolution: 640 x 640
- Compute: 4.6 GFLOPS
- Vitis AI Model Zoo version: 3.0

## Original upstream

AMD/Xilinx documentation identifies the original upstream project as:

Ultralytics YOLOv5

AMD/Xilinx states that it modified the upstream repository to create
its Vitis AI YOLOv5 fork.

The exact AMD/Xilinx modified training source used to produce this
specific XModel is not contained in this repository.

## Accuracy reported by AMD/Xilinx

- Official FLOAT AP[0.5:0.95]: 28.0%
- AMD FLOAT AP[0.5:0.95]: 27.0%
- AMD Quantized AP[0.5:0.95]: 26.2%

These are Model Zoo results and are not measurements from this KV260 project.

## Model preparation flow

Conceptually:

Ultralytics YOLOv5 source
        |
        v
AMD/Xilinx modified YOLOv5 fork
        |
        v
COCO-trained PyTorch model
        |
        v
Vitis AI quantization / QAT
        |
        v
Quantized model
        |
        v
Vitis AI compiler
        |
        +---- DPU architecture
        |
        v
yolov5_nano_pt.xmodel
        |
        v
DPUCZDX8G_ISA1_B4096
        |
        v
AMD/Xilinx Kria KV260

## Important project boundary

The exact training, QAT, and compilation steps for this pretrained Model
Zoo model were performed before the model package was distributed.

This project starts from the AMD/Xilinx supplied Vitis-AI-compatible
YOLOv5 Nano XModel and focuses on:

- KV260 DPU integration
- VART/XIR execution
- image preprocessing
- YOLO tensor decoding
- confidence filtering and NMS
- custom HLS Sobel acceleration
- heterogeneous PS/PL integration
- PetaLinux deployment
- CARLA/live-camera streaming
- benchmarking and validation

## KV260 deployment artifact

Model:

model/yolov5_nano_pt.xmodel

Target:

DPUCZDX8G_ISA1_B4096

VART runner:

libvart-dpu-runner.so

Expected MD5:

54934783deb174e4c782ce7c6bbcf27e

Verified project SHA256:

2a45d4dabd309cb8ebb52493139bb4b871980228e4904705e0ce28517bc0b646
