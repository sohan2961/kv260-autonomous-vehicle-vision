# KV260 Vision Applications

This directory contains three C++ applications for the AMD/Xilinx
Kria KV260 autonomous-vehicle vision project.

The applications integrate the custom HLS Sobel accelerator and
Vitis AI YOLOv5 Nano DPU.

## 1. Static Image Application

File: kv260-vision-app.cpp

Main functions:
- Load a static image using OpenCV.
- Execute the HLS Sobel accelerator.
- Execute YOLOv5 Nano inference through VART.
- Decode DPU output tensors.
- Generate processed results and timing measurements.

## 2. Video Application

File: kv260-vision-video-app.cpp

Main functions:
- Open video files or camera input using OpenCV.
- Process successive frames through the vision pipeline.
- Execute HLS Sobel and DPU inference.
- Generate visualization overlays.
- Record processed video using OpenCV VideoWriter.
- Support optional browser streaming.

## 3. Network Application

File: kv260-vision-network-app.cpp

This application extends the video pipeline with live TCP input.

The validated CARLA sender transmits frames as:

4-byte unsigned big-endian JPEG length
+
JPEG image payload

The application receives the payload and decodes it using
OpenCV imdecode.

Supported source modes include:
- TCP JPEG input
- Video file input
- Camera input

The application also supports AVI recording and optional
HTTP/MJPEG browser streaming.

## 4. Validated Live Demonstration

Run on the KV260:

    sudo /usr/bin/kv260-vision-network-app \
      /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel \
      --tcp 5000 \
      /tmp/kv260_carla_live.avi \
      --stream 8080

Open the processed output:

http://192.168.50.2:8080

## 5. Processing Responsibilities

HLS Sobel:
Hardware edge extraction.

Vitis AI DPU:
YOLOv5 Nano object detection.

ARM/OpenCV:
Preprocessing, output decoding, lane-candidate extraction,
visualization, video recording and streaming.

Sobel edge extraction alone is not semantic lane detection.

## 6. Deployment

The corresponding PetaLinux recipes are available under:

../petalinux/recipes-apps/

The validated prebuilt runtime is available under:

../v1/

See ../docs/QUICK_START.md for complete deployment instructions.
