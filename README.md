# Edge Traffic Counting Reproduction Bundle

Minimal standalone bundle for reproducing the edge-side vehicle counting
experiments and reference outputs from this project.

This folder intentionally contains only:

- the exact experiment scripts needed to rerun the benchmark
- the small configuration, mask, and model files required by those scripts
- reference result artifacts for result checking and summary regeneration

It intentionally does **not** vendor the three raw CCTV videos, because they are large:

- `7_45.mp4`
- `17_30.mp4`
- `18_45.mp4`

Place those three files in the repository root before re-running experiments.

## Layout

- `gt.csv`: ground-truth counts
- `roi_masks/`: roadway ROI masks
- `yolov8n.pt`, `yolo11n.pt`: detector weights used by the benchmark
- `models/zero_dce.torchscript.pt`: learned low-light enhancer used by `P3`
- `benchmark/run_benchmark.py`: main experiment runner
- `benchmark/generate_nvdcf_tracks.py`: DeepStream/NvDCF export helper
- `deepstream_backend.py`: DeepStream backend required by NvDCF export
- `monitor_power.py`: power logger used on Jetson
- `benchmark/results_full_power/`: reference Python/ByteTrack results
- `benchmark/reference_results_maxn/`: reference MAXN summary outputs
- `benchmark/reference_results_nvdcf_30w/`: reference NvDCF comparison outputs
- `scripts/`: small helper scripts for rerunning and summarizing results

## Hardware / Software Assumptions

The reference runs in this bundle target:

- NVIDIA Jetson AGX Orin
- `MAXN` power mode for the main throughput/latency runs
- JetPack 6.x
- TensorRT / DeepStream 7.1 for the NvDCF comparison path

Python dependencies used by the included scripts:

- `torch`
- `ultralytics`
- `supervision`
- `opencv-python`
- `numpy`
- `Pillow`
- `jetson-stats` (`jtop`) for power logging

DeepStream is required only for `generate_nvdcf_tracks.py`.

## Quick Start

1. Put the three raw videos in the repo root:

```bash
cp /path/to/7_45.mp4 .
cp /path/to/17_30.mp4 .
cp /path/to/18_45.mp4 .
```

2. Create a Python environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Reproduce the main Python/ByteTrack experiment block:

```bash
bash scripts/run_core_experiments.sh
```

4. If you also want the DeepStream NvDCF comparison:

```bash
bash scripts/run_nvdcf_comparison.sh
```

5. Summarize the copied or regenerated reference metrics:

```bash
python scripts/extract_reference_metrics.py
```

6. Regenerate compact summary tables from the bundled result artifacts:

```bash
python scripts/extract_reference_tables.py
```

You can also use the provided `Makefile`:

```bash
make metrics
make tables
make core
make nvdcf
```

## Exact Commands

### Main `R1`–`R5` experiment block

This command reproduces the main three-video Python experiment family:

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

Notes:

- This script expects the Jetson to already be in the desired power mode.
- If you want to switch power modes automatically, extend the command with the
  `--apply-nvpmodel` flags supported by `benchmark/run_benchmark.py`.

### NvDCF comparison path

First export NvDCF tracks through DeepStream:

```bash
python benchmark/generate_nvdcf_tracks.py --max-frames 0
```

Then evaluate those exported tracks in the same counting protocol:

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
  --output-dir ./benchmark/reference_results_nvdcf_30w \
  --max-frames 0 \
  --nmsfree-detector yolo11n \
  --enable-tracking \
  --tracker nvdcf \
  --external-tracks-template './benchmark/nvdcf_tracks/{run_id}_{POWER_MODE}_{condition}.txt'
```

## Reference Results Included

This bundle already includes the reference JSON/CSV outputs copied from the source
workspace. Those files are useful for:

- verifying that a fresh rerun matches the reference outputs
- checking the exact protocol metadata used for the reported runs
- regenerating compact result summaries without rerunning the full experiment

Use:

```bash
python scripts/extract_reference_metrics.py
```

to print a compact summary from the included artifacts.

Use:

```bash
python scripts/extract_reference_tables.py
```

to regenerate compact markdown tables directly from the copied JSON outputs.

Equivalent `Makefile` targets are:

- `make metrics`
- `make tables`
- `make core`
- `make nvdcf`

## Reproducibility Notes

- The benchmark expects the three source videos to keep the same filenames:
  `7_45.mp4`, `17_30.mp4`, and `18_45.mp4`.
- Detector weights, ROI masks, and ground-truth counts are bundled here so the
  experiment inputs are fixed.
- `P3` runs depend on `models/zero_dce.torchscript.pt`.
- The DeepStream path is only needed when reproducing the `NvDCF` comparison.
- Reference outputs are included so other users can compare a fresh rerun
  against known-good JSON summaries before extending the benchmark.

## Turning This Into a Git Repo

From inside this folder:

```bash
git init
git add .
git commit -m "Add edge traffic counting reproduction bundle"
```

The `.gitignore` intentionally excludes the large raw videos and generated outputs.
