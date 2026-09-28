# KV260 Custom PetaLinux Runtime

Target: AMD/Xilinx Kria KV260
Toolchain: PetaLinux 2022.2

This directory contains the custom Linux integration used by the
HLS Sobel and Vitis AI DPU autonomous-vehicle vision application.

## 1. Runtime Architecture

CARLA / Camera
      |
      v
Ethernet TCP
      |
      v
PetaLinux on KV260
      |
      +-- Vitis AI DPU driver
      +-- Custom Sobel DMA driver
      +-- VART / XIR
      +-- OpenCV vision applications
      |
      v
Live browser output + AVI recording

## 2. Permanent Ethernet

Configuration:

recipes-apps/kv260-runtime-config/files/10-eth0.network

KV260 address: 192.168.50.2/24
Host address:  192.168.50.1/24

DHCP is disabled.

The device-tree customization configures GEM3 and the
TI DP83867 Ethernet PHY.

## 3. Driver Initialization

Service:

kv260-vision-drivers.service

The service executes:

- modprobe xlnx_dpu
- modprobe sobel-dma-driver

It verifies the presence of:

- /dev/dpu
- /dev/sobel_accel

The service is enabled through the runtime BitBake recipe.

The repository also contains a modules-load configuration.
FPGA configuration must precede successful probing of the
PL-dependent drivers.

## 4. Sobel DMA Driver

Source:

recipes-modules/sobel-dma-driver/

The driver provides the Linux interface to the custom
HLS Sobel accelerator.

Its recipe contains a blacklist configuration to prevent
premature automatic loading through the device-tree modalias.

The runtime service explicitly loads the module.

## 5. Device Tree

Custom source:

recipes-bsp/device-tree/files/system-user.dtsi

It configures:

- GEM3 Ethernet
- TI DP83867 PHY
- PHY reset GPIO38
- MIO pin control
- UART console

The pl-custom.dtsi file is empty.

The hardware-generated pl.dtsi contains:

- DPU: 0x80000000, interrupt 89
- Sobel: 0xA0000000, interrupt 90

The generated device-tree snapshots are preserved under
documentation/device-tree/.

The custom system-user.dtsi provides board-specific Ethernet
and UART configuration.

## 6. Permanent Applications

The validated image contains:

- kv260-vision-app
- kv260-vision-video-app
- kv260-vision-network-app
- start-vision-demo
- YOLOv5 Nano xmodel

The permanent model path is:

/usr/share/kv260-vision/model/yolov5_nano_pt.xmodel

## 7. Live Demonstration

The start-vision-demo wrapper starts TCP reception on port 5000.

For the validated browser-enabled demonstration, execute:

sudo /usr/bin/kv260-vision-network-app \
  /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel \
  --tcp 5000 \
  /tmp/kv260_carla_live.avi \
  --stream 8080

Browser:

http://192.168.50.2:8080

## 8. Runtime Verification

Useful commands on the KV260:

systemctl status kv260-vision-drivers.service
ls -l /dev/dpu /dev/sobel_accel
xdputil query
ip -4 addr show eth0

Expected DPU fingerprint:

0x101000056010407

## 9. Validated Release

The prebuilt runtime is available in:

../v1/

The verified image.ub already contains the permanent
vision applications and YOLO model.

See ../docs/QUICK_START.md for deployment instructions.
