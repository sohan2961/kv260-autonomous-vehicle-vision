# KV260 Performance Benchmarks

This directory contains measured performance results from
the physical AMD/Xilinx Kria KV260.

## Validated Live CARLA Test

| Metric | Result |
|---|---:|
| Frames processed | 696 |
| HLS Sobel | 1.74246 ms |
| Preprocessing | 10.3329 ms |
| DPU execution | 7.24071 ms |
| Postprocessing | 1.97276 ms |
| Compute time per frame | 21.2888 ms |
| Compute FPS | 46.9731 |
| End-to-end time per frame | 97.1754 ms |
| End-to-end FPS | 10.2907 |

## Test Configuration

- Target board: Kria KV260
- Input: CARLA RGB camera
- Resolution: 640x480
- Target stream rate: 10 FPS
- Transport: TCP Ethernet
- Processing: HLS Sobel + YOLOv5 Nano DPU

## Compute FPS vs End-to-End FPS

Compute FPS represents the processing rate of the application
pipeline.

End-to-end FPS includes the live frame-transfer pipeline
and is limited by the approximately 10 FPS camera stream.

The end-to-end measurement excludes browser/HTTP delivery
and one-time initialization.

These two FPS measurements must not be interpreted as
equivalent performance metrics.

## Source Results

The original measured results are preserved in:

live_carla_benchmark.txt

The full demonstration videos are available in GitHub
Release v1.0.0.
