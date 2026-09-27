# Troubleshooting

## Cannot ping the KV260

Expected addresses:

```text
Host PC : 192.168.50.1/24
KV260   : 192.168.50.2/24
```

Windows:

```powershell
ping 192.168.50.2
```

If the host adapter is not `192.168.50.1/24`, configure it first.

## Browser shows `ERR_CONNECTION_REFUSED`

Port `8080` is not currently being served.

Use the explicit validated command on the KV260:

```bash
sudo /usr/bin/kv260-vision-network-app \
  /usr/share/kv260-vision/model/yolov5_nano_pt.xmodel \
  --tcp 5000 \
  /tmp/kv260_carla_live.avi \
  --stream 8080
```

Then open:

```text
http://192.168.50.2:8080
```

## CARLA sender cannot connect to port 5000

Start the KV260 network application first. It should report that it is listening for a TCP JPEG sender.

Then run:

```powershell
py -3.7 carla_kv260_live_v2.py
```

## `scp` reports missing `sftp-server`

The minimal PetaLinux image does not include `/usr/libexec/sftp-server`.

Use legacy SCP:

```powershell
scp -O petalinux@192.168.50.2:/home/petalinux/kv260_carla_live.avi .
```

## SSH says remote host identification changed

This can occur after replacing/rebuilding the PetaLinux image.

After verifying that `192.168.50.2` is your directly connected KV260:

```powershell
ssh-keygen -R 192.168.50.2
```

Reconnect and accept the new key.

## Verify the DPU

```bash
sudo xdputil query
```

Expected fingerprint:

```text
0x101000056010407
```

## `/dev/dpu` or `/dev/sobel_accel` is missing

Check the permanent driver service:

```bash
systemctl status kv260-vision-drivers.service
```

Also check:

```bash
lsmod | grep -E 'xlnx_dpu|sobel'
ls -l /dev/dpu /dev/sobel_accel
```

## Result video disappears after reboot

The default live output is under `/tmp`, which is temporary.

Copy important recordings before rebooting:

```bash
sudo cp /tmp/kv260_carla_live.avi /home/petalinux/
```

Then transfer it to the host PC.

## Git LFS files are only pointer text

Install Git LFS and retrieve the actual binaries:

```bash
git lfs install
git lfs pull
```

Then verify:

```bash
cd v1
sha256sum -c SHA256SUMS.txt
```
