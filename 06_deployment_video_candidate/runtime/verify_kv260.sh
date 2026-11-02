#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../model/yolov5_nano_pt"
MODEL="$MODEL_DIR/yolov5_nano_pt.xmodel"

EXPECTED_FP="0x101000056010407"

echo "========================================"
echo " KV260 DPU + YOLO Runtime Verification"
echo "========================================"

echo
echo "=== 1. Vitis AI tools ==="
command -v xdputil || {
    echo "ERROR: xdputil not found"
    exit 1
}

echo
echo "=== 2. DPU hardware query ==="
if xdputil query | tee "$SCRIPT_DIR/xdputil_query.txt"; then
    echo "DPU QUERY = OK"
else
    echo "DPU QUERY = FAILED"
    exit 1
fi

echo
echo "=== 3. Fingerprint check ==="
if grep -qi "$EXPECTED_FP" "$SCRIPT_DIR/xdputil_query.txt"; then
    echo "DPU FINGERPRINT MATCH = OK"
    echo "EXPECTED = $EXPECTED_FP"
else
    echo "DPU FINGERPRINT MATCH = FAILED"
    echo "EXPECTED = $EXPECTED_FP"
    exit 1
fi

echo
echo "=== 4. Model checksum ==="
EXPECTED_MD5="$(tr -d '[:space:]' < "$MODEL_DIR/md5sum.txt")"
ACTUAL_MD5="$(md5sum "$MODEL" | awk '{print $1}')"

echo "EXPECTED = $EXPECTED_MD5"
echo "ACTUAL   = $ACTUAL_MD5"

if [ "$EXPECTED_MD5" = "$ACTUAL_MD5" ]; then
    echo "MODEL CHECKSUM = OK"
else
    echo "MODEL CHECKSUM = FAILED"
    exit 1
fi

echo
echo "=== 5. XModel information ==="
if xdputil xmodel "$MODEL" -l | tee "$SCRIPT_DIR/xmodel_info.txt"; then
    echo "XMODEL READ = OK"
else
    echo "XMODEL READ = FAILED"
    exit 1
fi

echo
echo "=== 6. CMA memory ==="
grep -i '^Cma' /proc/meminfo || true

echo
echo "========================================"
echo " BASIC KV260 VALIDATION COMPLETE"
echo "========================================"
