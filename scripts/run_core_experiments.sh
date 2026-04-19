#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python benchmark/run_benchmark.py \
  --video-mode three \
  --video-day "$ROOT/7_45.mp4" \
  --video-dusk "$ROOT/17_30.mp4" \
  --video-night "$ROOT/18_45.mp4" \
  --roi-mask-day "$ROOT/roi_masks/7_45_roi.png" \
  --roi-mask-dusk "$ROOT/roi_masks/17_30_roi.png" \
  --roi-mask-night "$ROOT/roi_masks/18_45_roi.png" \
  --gt-count "$ROOT/gt.csv" \
  --output-dir "$ROOT/benchmark/results_full_power" \
  --max-frames 0 \
  --nmsfree-detector yolo11n \
  --strict-p3 \
  --zero-dce-weights "$ROOT/models/zero_dce.torchscript.pt" \
  --p3-lowlight-conditions dusk,night \
  --tau-night 0.35 \
  --enable-tracking \
  --tracker bytetrack \
  --with-power
