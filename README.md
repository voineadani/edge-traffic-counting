# Edge Traffic Counting

This repository contains the code, configuration files, masks, model weights,
and reference outputs needed to reproduce the vehicle-counting experiments in
this project on Jetson AGX Orin. The benchmark covers three fixed traffic
videos recorded under day, dusk, and night conditions, and includes both the
standard Python pipeline with ByteTrack and the DeepStream-based NvDCF
comparison path.

The repository includes the processed experiment outputs used as reference
artifacts, so a fresh run can be checked against known results without having
to reconstruct tables by hand.

## Data Access

The three raw CCTV videos are not distributed through GitHub. If you need them
for replication, contact `daniel.voinea@unitbv.ro`.

The benchmark expects the following filenames in the repository root:

- `7_45.mp4`
- `17_30.mp4`
- `18_45.mp4`

## Repository Contents

- `benchmark/run_benchmark.py`: main benchmark runner
- `benchmark/generate_nvdcf_tracks.py`: DeepStream helper for NvDCF track export
- `benchmark/benchmark_segments.template.json`: segment template for the three conditions
- `benchmark/results_full_power/`: bundled reference outputs for the Python pipeline
- `benchmark/reference_results_maxn/`: bundled MAXN reference outputs
- `benchmark/reference_results_nvdcf_30w/`: bundled NvDCF comparison outputs
- `scripts/extract_reference_metrics.py`: compact metric summary from bundled outputs
- `scripts/extract_reference_tables.py`: markdown table extraction from bundled outputs
- `scripts/run_core_experiments.sh`: one-command run for the main benchmark block
- `scripts/run_nvdcf_comparison.sh`: one-command run for the NvDCF comparison
- `roi_masks/`: ROI masks used for counting
- `gt.csv`: ground-truth vehicle totals
- `models/zero_dce.torchscript.pt`: Zero-DCE weights used by `P3`
- `yolov8n.pt`, `yolo11n.pt`: detector weights used in the benchmark

## Hardware and Software

The reference runs were produced on:

- NVIDIA Jetson AGX Orin
- `MAXN` mode for the main throughput and latency measurements
- JetPack 6.x
- DeepStream 7.1 for the NvDCF comparison path

Python packages used by the included scripts:

- `torch`
- `ultralytics`
- `supervision`
- `opencv-python`
- `numpy`
- `Pillow`
- `jetson-stats`

DeepStream is only required for `benchmark/generate_nvdcf_tracks.py` and the
`make nvdcf` path.

## Setup

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the three videos in the repository root:

```bash
cp /path/to/7_45.mp4 .
cp /path/to/17_30.mp4 .
cp /path/to/18_45.mp4 .
```

## Running the Benchmark

Run the main Python pipeline:

```bash
bash scripts/run_core_experiments.sh
```

Run the DeepStream NvDCF comparison:

```bash
bash scripts/run_nvdcf_comparison.sh
```

The same entry points are available through `make`:

```bash
make core
make nvdcf
```

## Inspecting the Bundled Outputs

Print a compact summary of the reference outputs:

```bash
python scripts/extract_reference_metrics.py
```

Generate markdown tables from the bundled outputs:

```bash
python scripts/extract_reference_tables.py
```

Or use:

```bash
make metrics
make tables
```

## Reference Commands

Main benchmark run:

```bash
python benchmark/run_benchmark.py \
  --video-mode three \
  --video-day ./7_45.mp4 \
  --video-dusk ./17_30.mp4 \
  --video-night ./18_45.mp4 \
  --roi-mask-day ./roi_masks/7_45_roi.png \
  --roi-mask-dusk ./roi_masks/17_30_roi.png \
  --roi-mask-night ./roi_masks/18_45_roi.png \
  --gt-count ./gt.csv \
  --output-dir ./benchmark/results_full_power \
  --max-frames 0 \
  --nmsfree-detector yolo11n \
  --strict-p3 \
  --zero-dce-weights ./models/zero_dce.torchscript.pt \
  --p3-lowlight-conditions dusk,night \
  --tau-night 0.35 \
  --enable-tracking \
  --tracker bytetrack \
  --with-power
```

NvDCF comparison:

```bash
python benchmark/generate_nvdcf_tracks.py --max-frames 0

python benchmark/run_benchmark.py \
  --video-mode three \
  --video-day ./7_45.mp4 \
  --video-dusk ./17_30.mp4 \
  --video-night ./18_45.mp4 \
  --roi-mask-day ./roi_masks/7_45_roi.png \
  --roi-mask-dusk ./roi_masks/17_30_roi.png \
  --roi-mask-night ./roi_masks/18_45_roi.png \
  --gt-count ./gt.csv \
  --output-dir ./benchmark/reference_results_nvdcf_30w \
  --max-frames 0 \
  --nmsfree-detector yolo11n \
  --enable-tracking \
  --tracker nvdcf \
  --external-tracks-template './benchmark/nvdcf_tracks/{run_id}_{POWER_MODE}_{condition}.txt'
```

## Notes

- The three videos must keep the filenames listed above.
- ROI masks, ground truth, model weights, and reference outputs are included so
  the benchmark can be rerun with the same inputs used here.
- `P3` depends on `models/zero_dce.torchscript.pt`.
- The `.gitignore` excludes the raw videos and generated local outputs.
