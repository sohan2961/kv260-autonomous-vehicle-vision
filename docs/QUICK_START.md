# Quick Start — KV260 Autonomous Vehicle Vision

This is the validated **easy demo path**. It uses the prebuilt KV260 runtime and does not require rebuilding Vivado or PetaLinux.

## 1. Clone the repository

```bash
git clone https://github.com/sohan2961/kv260-autonomous-vehicle-vision.git
cd kv260-autonomous-vehicle-vision
```

The large files under `v1/` use Git LFS:

```bash
git lfs install
git lfs pull
```

Alternatively, download the same validated binaries from the GitHub `v1.0.0` Release.

## 2. Verify the validated v1 artifacts

```bash
cd v1
sha256sum -c SHA256SUMS.txt
```

Expected:

```text
BOOT.BIN: OK
boot.scr: OK
image.ub: OK
system.bit: OK
```

Validated hashes:

```text
a75156bdd7dbc718a10090e835fd836a98779c341d1aa2e7f8c0715930953499  BOOT.BIN
03c6e6b965bbfbfb7fdec4bf59fedef8d9a43dbc1a758fb5d935b52be5945bb9  boot.scr
c25df88ae6ad5334738f18fb9d00118404c30de4c45b724424d9e8b759d7dd40  image.ub
6450eee7649e990807270896642bf30b211851796f1d8b0cd0f33bb5f4f908e4  system.bit
```

## 3. Prepare the SD card

Copy these files to the prepared KV260 FAT boot partition:

```text
BOOT.BIN
boot.scr
image.ub
system.bit
```

Insert the SD card into the KV260.

The validated `image.ub` already contains the YOLOv5 Nano model and KV260 applications.

## 4. Connect Ethernet

Use a direct Ethernet connection:

```text
Host PC / laptop : 192.168.50.1/24
KV260            : 192.168.50.2/24
```

The KV260 address is configured automatically by the validated runtime.

From Windows:

```powershell
ping 192.168.50.2
```

## 5. Boot and verify the KV260

Power on the KV260 and wait for PetaLinux to boot.

The validated runtime automatically loads the DPU and Sobel drivers.

Optional DPU check:

```bash
sudo xdputil query
```

Expected fingerprint:

```text
0x101000056010407
```

## 6. Start the live KV260 application

For the exact browser-enabled configuration validated in this project, run on the KV260:

```bash
sudo /usr/bin/kv260-vision-network-app \
  /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel \
  --tcp 5000 \
  /tmp/kv260_carla_live.avi \
  --stream 8080
```

Expected ports:

```text
TCP CARLA input : 5000
HTTP/MJPEG      : 8080
KV260 IP        : 192.168.50.2
```

`sudo start-vision-demo` is also installed in the runtime, but the command above explicitly enables the browser stream and therefore documents the exact validated live-demo invocation.

## 7. Start CARLA on Windows

Validated environment:

```text
CARLA       : 0.9.9.2
Python      : 3.7
RGB camera  : 640×480
Target rate : 10 FPS
```

Start the CARLA simulator before running the sender.

## 8. Run the CARLA V2 sender

From the cloned repository on the CARLA host:

```powershell
cd carla
py -3.7 carla_kv260_live_v2.py
```

The sender targets:

```text
192.168.50.2:5000
```

The V2 script adds traffic vehicles and pedestrians so the KV260 DPU has useful detection targets.

**Detection responsibility**

- Car/person/object boxes: **KV260 YOLOv5 Nano DPU**
- Intersection status: **CARLA map ground truth**, not YOLO

## 9. Open the live processed result

On the host PC/laptop open:

```text
http://192.168.50.2:8080
```

Pipeline:

```text
CARLA RGB
   ↓
TCP Ethernet
   ↓
KV260
 ├─ HLS Sobel
 └─ YOLOv5 Nano DPU
   ↓
ARM/OpenCV
   ↓
Live split-screen + AVI recording
```

## 10. Save the AVI result before rebooting

The output is written to:

```text
/tmp/kv260_carla_live.avi
```

`/tmp` is temporary. Copy the result before powering off:

```bash
sudo cp /tmp/kv260_carla_live.avi /home/petalinux/kv260_carla_live.avi
sudo chown petalinux:petalinux /home/petalinux/kv260_carla_live.avi
```

Copy to Windows with legacy SCP mode:

```powershell
scp -O petalinux@192.168.50.2:/home/petalinux/kv260_carla_live.avi "C:\Users\YOUR_NAME\Desktop\KV260_Results\kv260_carla_live.avi"
```

## Validated Live Benchmark

```text
Frames processed         = 696
Average HLS Sobel        = 1.74246 ms
Average preprocessing    = 10.3329 ms
Average DPU execution    = 7.24071 ms
Average postprocessing   = 1.97276 ms
Average compute/frame    = 21.2888 ms
Average compute FPS      = 46.9731
Average end-to-end/frame = 97.1754 ms
Average end-to-end FPS   = 10.2907
```

End-to-end timing excludes preview/HTTP delivery and one-time setup.
