# CARLA Live Simulation — KV260

This directory contains the validated CARLA live sender used to test
the FPGA-based autonomous-vehicle vision pipeline.

## 1. Tested Environment

| Component | Configuration |
|---|---|
| Simulator | CARLA 0.9.9.2 |
| Operating system | Windows |
| Python | 3.7 |
| RGB resolution | 640 × 480 |
| Target camera rate | 10 FPS |
| JPEG quality | 85 |
| Additional vehicles | 12 |
| Pedestrians | 12 |

## 2. Processing Architecture

```text
CARLA Simulator (Windows)
          |
          v
RGB Camera 640×480
          |
          v
JPEG Encoding
          |
          v
TCP Ethernet
          |
          v
AMD Kria KV260
    /           \
   v             v
HLS Sobel     YOLOv5 Nano DPU
   \             /
    \           /
     v         v
      ARM/OpenCV
          |
          v
Live split-screen result
     /           \
    v             v
Browser :8080   AVI Recording
```

## 3. Network Configuration

| Device | IP address |
|---|---|
| Windows Ethernet | 192.168.50.1/24 |
| KV260 | 192.168.50.2/24 |

CARLA server: localhost:2000

KV260 TCP input: 192.168.50.2:5000

Browser output: http://192.168.50.2:8080

## 4. TCP Frame Protocol

Each JPEG frame is transmitted as:

```text
[4-byte unsigned big-endian JPEG length]
[JPEG image payload]
```

The KV260 network application receives and decodes each frame.

## 5. Start the KV260 Application

On the KV260:

```bash
sudo /usr/bin/kv260-vision-network-app \
  /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel \
  --tcp 5000 \
  /tmp/kv260_carla_live.avi \
  --stream 8080
```

## 6. Run the CARLA Sender

Start CARLA 0.9.9.2 on Windows.

From the repository's carla directory:

```powershell
py -3.7 carla_kv260_live_v2.py
```

Note: the original script contains a machine-specific CARLA_ROOT
path. Other users must configure their local CARLA installation
accordingly.

## 7. Simulation Features

The V2 sender includes:

- Additional NPC vehicles
- Additional pedestrians
- Ego vehicle autopilot
- Junction-oriented simulation setup
- Live RGB camera streaming
- CARLA map-based intersection indication

IMPORTANT:

Vehicle and pedestrian detection is performed by the KV260
YOLOv5 Nano DPU.

The intersection indicator is obtained from CARLA map ground
truth and is labelled [CARLA MAP GT].

It is not an intersection detected by YOLO.

## 8. Validated Live Benchmark

| Metric | Result |
|---|---:|
| Processed frames | 696 |
| HLS Sobel | 1.74246 ms |
| DPU execution | 7.24071 ms |
| Compute FPS | 46.9731 |
| End-to-end FPS | 10.2907 |

End-to-end timing excludes browser/HTTP delivery and one-time setup.

See ../benchmarks/live_carla_benchmark.txt.

## 9. Output

Browser:

http://192.168.50.2:8080

AVI recording:

/tmp/kv260_carla_live.avi

Copy the AVI to permanent storage before rebooting.
