# Architecture

## End-to-End Data Flow

```text
Windows PC / CARLA 0.9.9.2
          |
          | RGB 640×480 @ ~10 FPS
          | JPEG + 4-byte big-endian length
          v
Direct Ethernet
Host: 192.168.50.1
KV260: 192.168.50.2
          |
          v
kv260-vision-network-app
          |
     +----+-------------------+
     |                        |
     v                        v
Custom HLS Sobel        Vitis AI DPU
edge accelerator        YOLOv5 Nano
     |                        |
     +-----------+------------+
                 |
                 v
          ARM Cortex-A53
           OpenCV logic
                 |
                 v
          1280×520 composite
            /           \
           v             v
 HTTP/MJPEG :8080    AVI recording
```

## Responsibilities

### HLS Sobel accelerator

Performs hardware edge extraction. It is not itself a semantic lane detector.

### Vitis AI DPU

Runs the YOLOv5 Nano model for COCO object detection, including people, cars, buses, trucks, bicycles, traffic lights, and other supported classes.

### ARM/OpenCV

Handles frame preparation, YOLO post-processing, visualization, composition, timing, and lane-related post-processing/overlay.

### CARLA V2 sender

Supplies simulated live RGB frames and adds traffic/pedestrian actors for testing. Its intersection indicator uses CARLA map ground truth and must not be interpreted as a YOLO-detected intersection.

## Network Protocol

Each frame sent from the CARLA host to the KV260 uses:

```text
4-byte unsigned big-endian JPEG length
+
JPEG payload
```

The KV260 listens on TCP port `5000`.

The processed browser stream is exposed on HTTP/MJPEG port `8080`.

## Validated Network

```text
Windows host : 192.168.50.1/24
KV260        : 192.168.50.2/24
```

## Runtime Components

The permanent PetaLinux runtime includes:

- DPU driver
- Sobel DMA driver
- `kv260-vision-app`
- `kv260-vision-video-app`
- `kv260-vision-network-app`
- YOLOv5 Nano xmodel
- static Ethernet configuration
- automatic driver-loading service
