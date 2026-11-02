#!/bin/sh

set -u

DEPLOY_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

MODEL="$DEPLOY_DIR/model/yolov5_nano_pt/yolov5_nano_pt.xmodel"
IMAGE="$DEPLOY_DIR/test/road_test.jpg"
RESULTS="$DEPLOY_DIR/results"
LOG="$RESULTS/first_test.log"

echo "========================================"
echo " KV260 DPU + HLS Sobel First Test"
echo "========================================"
echo

if [ ! -f "$MODEL" ]; then
    echo "ERROR: XModel not found:"
    echo "$MODEL"
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: test image not found:"
    echo "$IMAGE"
    exit 1
fi

if [ ! -x /usr/bin/kv260-vision-app ]; then
    echo "ERROR: /usr/bin/kv260-vision-app not found"
    exit 1
fi

if [ ! -e /dev/sobel_accel ]; then
    echo "ERROR: /dev/sobel_accel does not exist"
    echo
    echo "Check:"
    echo "  lsmod | grep sobel"
    echo "  dmesg | grep -i sobel"
    exit 1
fi

mkdir -p "$RESULTS"

echo "Model : $MODEL"
echo "Image : $IMAGE"
echo "Output: $RESULTS"
echo

cd "$RESULTS" || exit 1

if /usr/bin/kv260-vision-app \
    "$MODEL" \
    "$IMAGE" \
    >"$LOG" 2>&1
then
    cat "$LOG"

    echo
    echo "========================================"
    echo " TEST COMPLETED"
    echo "========================================"

    ls -lh \
        "$RESULTS/result.jpg" \
        "$RESULTS/sobel_result.jpg" \
        2>/dev/null

    exit 0
else
    RC=$?

    cat "$LOG"

    echo
    echo "ERROR: vision application failed"
    echo "Exit code: $RC"

    exit "$RC"
fi
