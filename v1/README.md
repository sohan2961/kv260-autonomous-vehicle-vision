# Validated Prebuilt Runtime — v1

This directory contains the exact validated prebuilt runtime used for the physical KV260 live CARLA demonstration.

## Files

```text
BOOT.BIN
boot.scr
image.ub
system.bit
SHA256SUMS.txt
```

Large binaries are stored with Git LFS.

After cloning:

```bash
git lfs install
git lfs pull
```

## Verify Before Use

```bash
cd v1
sha256sum -c SHA256SUMS.txt
```

Expected hashes:

```text
a75156bdd7dbc718a10090e835fd836a98779c341d1aa2e7f8c0715930953499  BOOT.BIN
03c6e6b965bbfbfb7fdec4bf59fedef8d9a43dbc1a758fb5d935b52be5945bb9  boot.scr
c25df88ae6ad5334738f18fb9d00118404c30de4c45b724424d9e8b759d7dd40  image.ub
6450eee7649e990807270896642bf30b211851796f1d8b0cd0f33bb5f4f908e4  system.bit
```

## Runtime Network

```text
Host PC : 192.168.50.1/24
KV260   : 192.168.50.2/24
TCP     : 5000
HTTP    : 8080
```

## Expected DPU Fingerprint

```text
0x101000056010407
```

## Live Demo

For the exact validated browser-enabled invocation:

```bash
sudo /usr/bin/kv260-vision-network-app \
  /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel \
  --tcp 5000 \
  /tmp/kv260_carla_live.avi \
  --stream 8080
```

Then run the CARLA sender on Windows and open:

```text
http://192.168.50.2:8080
```

See [`../docs/QUICK_START.md`](../docs/QUICK_START.md) for the complete procedure.

The same files are also attached to GitHub Release `v1.0.0`.
