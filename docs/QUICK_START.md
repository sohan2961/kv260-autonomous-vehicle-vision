# Quick Start — KV260 Autonomous Vehicle Vision

This guide is the easy demonstration path for running the project without rebuilding the full Vivado/PetaLinux toolchain.

## 1. Clone the repository

```bash
git clone https://github.com/sohan2961/kv260-autonomous-vehicle-vision.git
cd kv260-autonomous-vehicle-vision
```

## 2. Download the prebuilt release

Download the validated prebuilt KV260 release from the GitHub Releases page.

The release bundle should contain:

```text
BOOT.BIN
boot.scr
image.ub
system.bit
```

> Note: in the validated permanent runtime image, the YOLOv5 Nano model is also installed inside the PetaLinux root filesystem at:
>
>
> Therefore a separate `.xmodel` file is only needed if you want to replace or inspect the model independently.

## 3. Prepare the SD card

Copy the prebuilt boot files to the prepared KV260 SD card boot partition:

```text
BOOT.BIN
boot.scr
image.ub
system.bit
```

Insert the SD card into the KV260.

## 4. Connect the laptop/PC to the KV260

Connect the laptop or Windows PC directly to the KV260 with Ethernet.

```text
Laptop/PC Ethernet : 192.168.50.1/24
KV260               : 192.168.50.2/24
```

The KV260 address is configured automatically by the permanent runtime.

Check connectivity:

```powershell
ping 192.168.50.2
```

## 5. Boot the KV260

Power on the KV260 and wait for PetaLinux to finish booting.

The validated image automatically loads the DPU and Sobel drivers.

Expected DPU fingerprint:

```text
0x101000056010407
```

## 6. Start the KV260 vision application

On the KV260:

```bash
sudo start-vision-demo
```

The intended demo configuration is:

```text
TCP input port : 5000
HTTP/MJPEG     : 8080
KV260 IP       : 192.168.50.2
```

If the installed `start-vision-demo` wrapper does not yet include browser streaming, start the network application explicitly with `--stream 8080`.

## 7. Start CARLA on the Windows computer

Validated environment:

```text
CARLA 0.9.9.2
Python 3.7
RGB camera: 640x480
Target rate: 10 FPS
```

Start the CARLA simulator first.

## 8. Run the live CARLA sender

From the cloned repository:

```powershell
cd carla
py -3.7 carla_kv260_live_v2.py
```

The sender streams JPEG frames over TCP to:

```text
192.168.50.2:5000
```

The V2 sender creates additional traffic vehicles and pedestrians.

Car/person detection is performed by the **KV260 YOLOv5 Nano DPU**.

Intersection status is derived from **CARLA map ground truth** and is not a YOLO class.

## 9. Open the live processed output

In a browser on the laptop/PC open:

```text
http://192.168.50.2:8080
```

Pipeline:

```text
CARLA
  ↓
Ethernet TCP
  ↓
KV260
 ├── HLS Sobel
 └── YOLOv5 Nano DPU
  ↓
ARM/OpenCV processing
  ↓
Live split screen
  ↓
Browser + AVI recording
```

## 10. Recorded result

The processed split-screen AVI is written on the KV260.

Example:

```text
/tmp/kv260_carla_live.avi
```

Because `/tmp` is temporary, copy important results before rebooting:

```bash
sudo cp /tmp/kv260_carla_live.avi /home/petalinux/kv260_carla_live.avi
sudo chown petalinux:petalinux /home/petalinux/kv260_carla_live.avi
```

On Windows, this PetaLinux image requires legacy SCP mode because it does not include the SFTP server:

```powershell
scp -O petalinux@192.168.50.2:/home/petalinux/kv260_carla_live.avi "C:\Users\YOUR_NAME\Desktop\KV260_Results\kv260_carla_live.avi"
```

## Validated live benchmark

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

End-to-end timing excludes browser/HTTP delivery and one-time setup.

## Build-from-source path

The full source tree contains the HLS Sobel accelerator, C++ vision applications, PetaLinux recipes, device-tree changes, DPU kernel patch, and Sobel DMA driver.

Matching toolchain versions:

```text
Vivado 2022.2
Vitis / Vitis HLS 2022.2
PetaLinux 2022.2
Vitis AI 3.0
Ubuntu 20.04
```
