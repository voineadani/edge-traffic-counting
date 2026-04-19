#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python benchmark/generate_nvdcf_tracks.py --max-frames 0

python benchmark/run_benchmark.py \
  --video-mode three \
  --video-day "$ROOT/7_45.mp4" \
  --video-dusk "$ROOT/17_30.mp4" \
  --video-night "$ROOT/18_45.mp4" \
  --roi-mask-day "$ROOT/roi_masks/7_45_roi.png" \
  --roi-mask-dusk "$ROOT/roi_masks/17_30_roi.png" \
  --roi-mask-night "$ROOT/roi_masks/18_45_roi.png" \
  --gt-count "$ROOT/gt.csv" \
  --output-dir "$ROOT/benchmark/reference_results_nvdcf_30w" \
  --max-frames 0 \
  --nmsfree-detector yolo11n \
  --enable-tracking \
  --tracker nvdcf \
  --external-tracks-template "$ROOT/benchmark/nvdcf_tracks/{run_id}_{POWER_MODE}_{condition}.txt"
