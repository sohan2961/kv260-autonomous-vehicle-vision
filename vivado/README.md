# Vivado Hardware Implementation

Target: AMD/Xilinx Kria KV260
Toolchain: Vivado 2022.2

## Hardware

- hardware/system.bit — validated FPGA bitstream
- hardware/kv260_dpu_sobel_eth.xsa — exported hardware platform

The bitstream is byte-identical to v1/system.bit.

## Timing Closure

Final routed implementation:

- WNS: +0.006 ns
- TNS: 0.000 ns
- WHS: +0.010 ns
- THS: 0.000 ns

All user-specified timing constraints are met.

## Reports

reports/synthesis contains synthesis utilization.

reports/implementation contains the available placement,
routing, timing, utilization, power, and DRC reports.

These reports were obtained from the completed implementation
preserved in the September 14 project backup.

## Constraints

constraints/sobel_accel_ooc.xdc is an IP-level
out-of-context constraint file.

No standalone top-level user XDC was found.

## Reproducibility

A final routed DCP and canonical Vivado recreation Tcl
are not currently included.

The validated implementation must not be replaced by a
newly generated bitstream without independent validation.

See SHA256SUMS.txt for artifact integrity verification.
