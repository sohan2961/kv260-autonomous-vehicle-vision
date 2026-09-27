# Vivado Reproducibility

The validated project used Vivado 2022.2 with a DPU + custom HLS Sobel design for the Kria KV260.

A canonical `recreate_design.tcl` is **not yet included** because it must be exported from the proven final Vivado design. A hand-written or newly regenerated script should not be presented as equivalent to the validated bitstream.

The validated prebuilt `system.bit` is available under `v1/` and in GitHub Release `v1.0.0`.

Future source-to-bitstream reproducibility work should export the known-good design to:

```text
vivado/recreate_design.tcl
```
