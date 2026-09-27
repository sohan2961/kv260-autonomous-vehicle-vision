# Build From Source

## Validated Toolchain

```text
Host OS              : Ubuntu 20.04 LTS
Vivado               : 2022.2
Vitis / Vitis HLS    : 2022.2
PetaLinux            : 2022.2
Vitis AI             : 3.0
Target               : AMD/Xilinx Kria KV260
```

## High-Level Build Flow

```text
HLS Sobel source
      ↓
Vitis HLS synthesis / co-simulation
      ↓
Vivado: DPU + Sobel integration
      ↓
Timing closure / bitstream / XSA
      ↓
PetaLinux 2022.2
  ├─ device tree
  ├─ DPU support
  ├─ Sobel DMA driver
  ├─ VART/XIR
  └─ vision applications
      ↓
BOOT.BIN + boot.scr + image.ub + system.bit
```

## Source Included in This Repository

- `hls/sobel/`: custom HLS Sobel source and testbench
- `application/`: C++ KV260 applications
- `petalinux/`: custom meta-user recipes/configuration
- `petalinux/recipes-bsp/device-tree/`: device-tree customizations
- `petalinux/recipes-kernel/`: DPU fingerprint-related kernel patch/configuration
- `petalinux/recipes-modules/sobel-dma-driver/`: custom Sobel DMA driver
- `carla/`: Windows live CARLA sender

## Important Reproducibility Note

The repository does **not yet contain a proven `recreate_design.tcl` exported from the final Vivado project**. Large generated Vivado/PetaLinux build directories are intentionally excluded from Git.

Do not invent or regenerate a different bitstream and assume it is equivalent to the validated release. The validated prebuilt hardware/runtime artifacts are provided in `v1/` and Release `v1.0.0`.

## Validated Prebuilt Artifacts

Use:

```bash
cd v1
sha256sum -c SHA256SUMS.txt
```

Expected DPU fingerprint at runtime:

```text
0x101000056010407
```

## Future Reproducibility Work

For complete source-to-bitstream reproducibility, export a Tcl script from the proven Vivado design and add it under:

```text
vivado/recreate_design.tcl
```

Only a Tcl export generated from the known-good design should be published as the canonical recreation script.
