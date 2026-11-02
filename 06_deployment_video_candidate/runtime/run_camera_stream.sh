#!/bin/sh

set -u

DEPLOY_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

MODEL="$DEPLOY_DIR/model/yolov5_nano_pt/yolov5_nano_pt.xmodel"
APP="/usr/bin/kv260-vision-video-app"

CAMERA_INDEX="${1:-0}"
STREAM_PORT="${2:-8080}"

RESULT_DIR="$DEPLOY_DIR/results/live_camera"
OUTPUT="$RESULT_DIR/live_camera_output.avi"

echo "========================================"
echo " KV260 LIVE HLS + DPU ROAD PERCEPTION"
echo "========================================"
echo "Model       : $MODEL"
echo "Camera      : /dev/video$CAMERA_INDEX"
echo "Output      : $OUTPUT"
echo "Stream port : $STREAM_PORT"
echo "========================================"

if [ ! -x "$APP" ]; then
    echo "ERROR: $APP not found"
    echo "Run this script on the KV260, not the host PC."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: XModel not found:"
    echo "$MODEL"
    exit 1
fi

if [ ! -e /dev/sobel_accel ]; then
    echo "ERROR: /dev/sobel_accel not found"
    exit 1
fi

if [ ! -e "/dev/video$CAMERA_INDEX" ]; then
    echo "ERROR: /dev/video$CAMERA_INDEX not found"
    echo
    echo "Available cameras:"
    ls -l /dev/video* 2>/dev/null || true
    exit 1
fi

mkdir -p "$RESULT_DIR"

echo
echo "Starting live perception..."
echo
echo "On the laptop open:"
echo "  http://<KV260-IP>:$STREAM_PORT/"
echo
echo "Press Ctrl+C to stop cleanly."
echo

cd "$RESULT_DIR" || exit 1

exec "$APP" \
    "$MODEL" \
    --camera "$CAMERA_INDEX" \
    "$OUTPUT" \
    --stream "$STREAM_PORT"
