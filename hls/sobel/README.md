# Custom HLS Sobel Accelerator

Target: AMD/Xilinx Kria KV260
Toolchain: Vitis HLS 2022.2
Target device: xck26-sfvc784-2LV-c

## Architecture

The custom Sobel accelerator performs hardware edge extraction.

It is integrated with the Vitis AI DPU in the KV260 Vivado design.
ARM/OpenCV handles lane-related post-processing and visualization.

## Source Files

- sobel_accel.cpp
- sobel_accel.h
- sobel_tb.cpp

The archived sobel_accel_fullhd.cpp and sobel_tb_fullhd.cpp
are byte-identical to their corresponding original files.

## HLS Synthesis Results

Target clock: 5.000 ns
Estimated clock period: 3.650 ns

Resources:

| Resource | Estimated usage |
|---|---:|
| LUT | 5825 |
| FF | 3134 |
| BRAM_18K | 2 |
| DSP | 11 |
| URAM | 0 |

These are HLS synthesis estimates, not final Vivado utilization.

## RTL Co-Simulation

Archived test: 320x240
RTL language: VHDL
Simulator: XSIM
Status: PASS
Latency: 86772 clock cycles

The archived co-simulation report does not establish
Full-HD RTL verification.

## Reports

reports/synthesis/ contains the final V2 synthesis report
and machine-readable synthesis XML.

reports/cosimulation/ contains the archived passing
320x240 VHDL co-simulation report.

## Runtime Validation

The accelerator was also tested on the physical KV260.

Static benchmark:
CPU Sobel: 13.8747 ms
HLS Sobel: 4.75452 ms
Speedup: approximately 2.92x

Live CARLA test:
Average HLS Sobel: 1.74246 ms

Runtime measurements and HLS synthesis estimates
represent different measurement conditions.
