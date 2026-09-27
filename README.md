# KV260 Autonomous Vehicle Vision

Real-time autonomous vehicle perception on the AMD/Xilinx Kria KV260.

## Main Features
- Custom HLS Sobel edge accelerator
- Vitis AI DPU with YOLOv5 Nano
- PetaLinux 2022.2
- Live CARLA 0.9.9.2 camera streaming over Ethernet
- YOLO object detection for vehicles and pedestrians
- HLS Sobel edge processing
- Live split-screen browser output
- AVI result recording

## Validated Performance
- Frames processed: 696
- HLS Sobel: 1.74246 ms
- DPU execution: 7.24071 ms
- Compute FPS: 46.9731
- End-to-end FPS: 10.2907

## Hardware
- AMD/Xilinx Kria KV260

## Software
- Vivado / Vitis / Vitis HLS 2022.2
- PetaLinux 2022.2
- Vitis AI 3.0
- OpenCV
- CARLA 0.9.9.2
- Python 3.7
