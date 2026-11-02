#!/bin/sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="$ROOT_DIR/model/yolov5_nano_pt/yolov5_nano_pt.xmodel"
IMAGE="$ROOT_DIR/test/road_test.jpg"

RESULT_DIR="$ROOT_DIR/results/static_benchmark"

APP="/usr/bin/kv260-vision-app"

WARMUP_RUNS=2
MEASURED_RUNS=10

echo "========================================"
echo " KV260 STATIC IMAGE BENCHMARK"
echo "========================================"
echo "Model   : $MODEL"
echo "Image   : $IMAGE"
echo "Warmups : $WARMUP_RUNS"
echo "Runs    : $MEASURED_RUNS"
echo "========================================"

if [ ! -x "$APP" ]; then
    echo "ERROR: $APP not found or not executable"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: XModel not found"
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: test image not found"
    exit 1
fi

if [ ! -e /dev/sobel_accel ]; then
    echo "ERROR: /dev/sobel_accel not found"
    exit 1
fi

mkdir -p "$RESULT_DIR"
cd "$RESULT_DIR" || exit 1

rm -f benchmark_results.csv

echo
echo "Running warm-up passes..."

i=1
while [ "$i" -le "$WARMUP_RUNS" ]; do

    echo "Warm-up $i/$WARMUP_RUNS"

    "$APP" "$MODEL" "$IMAGE" \
        > "warmup_${i}.log" 2>&1

    status=$?

    if [ "$status" -ne 0 ]; then
        echo "ERROR: warm-up $i failed"
        cat "warmup_${i}.log"
        exit "$status"
    fi

    i=$((i + 1))
done

# Remove warm-up measurements.
rm -f benchmark_results.csv

echo
echo "Running measured passes..."

i=1
while [ "$i" -le "$MEASURED_RUNS" ]; do

    echo "Measured run $i/$MEASURED_RUNS"

    "$APP" "$MODEL" "$IMAGE" \
        > "run_${i}.log" 2>&1

    status=$?

    if [ "$status" -ne 0 ]; then
        echo "ERROR: measured run $i failed"
        cat "run_${i}.log"
        exit "$status"
    fi

    i=$((i + 1))
done

echo
echo "========================================"
echo " BENCHMARK COMPLETE"
echo "========================================"
echo "Results:"
echo "$RESULT_DIR/benchmark_results.csv"
echo
echo "Rows:"
wc -l benchmark_results.csv
echo
echo "CSV preview:"
head -n 4 benchmark_results.csv
echo "========================================"
