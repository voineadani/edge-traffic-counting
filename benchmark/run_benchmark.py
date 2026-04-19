#!/usr/bin/env python3
"""
Benchmark experiment runner.

Implements the traffic-counting benchmark protocol:
- Visual conditions: day, dusk, night
- Preprocessing: P0 (Raw), P1 (CLAHE), P3 (condition-gated learned enhancement)
  - P3 gating: luminance threshold and/or condition gate for low-light segments
- Detectors: compact benchmark set
  - R1: YOLOv8n + P0
  - R2: YOLOv10n/YOLOv11n + P0 (selector flag)
  - R3: YOLOv8n + P1
  - R4: YOLOv8n + P3
  - R5: YOLOv10n/YOLOv11n + P3 (selector flag)
- Power modes: MAXN, 50W, 30W
- Timing buckets from median-normalized PTS:
  - Stable:  dt <= 1.25 * median_dt
  - Stutter: 1.25 * median_dt < dt <= 2.5 * median_dt
  - Drop:    dt > 2.5 * median_dt

Notes:
- This script is designed to match the experiment conditions and output schema.
- Learned enhancement models are pluggable. In day/dusk/night mode, P3 uses Zero-DCE.
- Default enhancer wrappers are provided under `benchmark/enhancers/` and support
  TorchScript models.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_VIDEO = REPO_ROOT / "faget.mp4"
DEFAULT_VIDEO_DAY = REPO_ROOT / "7_45.mp4"
DEFAULT_VIDEO_DUSK = REPO_ROOT / "17_30.mp4"
DEFAULT_VIDEO_NIGHT = REPO_ROOT / "18_45.mp4"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"
DEFAULT_SEGMENTS = SCRIPT_DIR / "benchmark_segments.template.json"

DETECTOR_PATHS = {
    "yolov8n": [REPO_ROOT / "yolov8n.pt"],
    # Prefer PT first for broad compatibility; engine/onnx are optional fallbacks.
    "yolov10n": [
        REPO_ROOT / "yolov10n.pt",
        REPO_ROOT / "yolo10n.pt",
        REPO_ROOT / "yolov10n.onnx_b1_gpu0_fp16.engine",
        REPO_ROOT / "yolov10n.onnx",
    ],
    # Prefer PT first for broad compatibility; engine/onnx are optional fallbacks.
    "yolo11n": [
        REPO_ROOT / "yolo11n.pt",
        REPO_ROOT / "yolo11n.onnx_b1_gpu0_fp16.engine",
        REPO_ROOT / "yolo11n.onnx",
    ],
}

VISUAL_CONDITIONS = ["day", "dusk", "night"]
POWER_MODES = ["MAXN"]
PREPROC_MODES = ["P0", "P1", "P3"]
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO vehicle classes: car, motorcycle, bus, truck

# Optional simulation when hardware power switching is not available.
SIMULATED_POWER_BUDGET_MS = {
    "MAXN": 33.0,
    "50W": 50.0,
    "30W": 80.0,
}

def build_core_runs(nmsfree_detector: str) -> List[Dict[str, str]]:
    return [
        {"run_id": "R1", "detector": "yolov8n", "preproc": "P0"},
        {"run_id": "R2", "detector": nmsfree_detector, "preproc": "P0"},
        {"run_id": "R3", "detector": "yolov8n", "preproc": "P1"},
        {"run_id": "R4", "detector": "yolov8n", "preproc": "P3"},
        {"run_id": "R5", "detector": nmsfree_detector, "preproc": "P3"},
    ]


@dataclass
class Segment:
    condition: str
    start_sec: float
    end_sec: float


@dataclass
class FrameRecord:
    frame_idx: int
    pts_ms: float
    dt_ms: float
    visual_condition: str
    timing_bucket: str
    preproc_latency_ms: float
    infer_latency_ms: float
    tracking_latency_ms: float
    latency_ms: float
    num_detections: int
    mean_confidence: float
    p3_enhanced: bool = False
    mean_luminance: float = 0.0


@dataclass
class RunSummary:
    run_id: str
    video: str
    video_condition: str
    detector: str
    detector_path: str
    preproc: str
    power_mode: str
    frames_processed: int
    wall_time_sec: float
    fps: float
    fps_per_watt: float
    avg_power_mw: float
    median_dt_ms: float
    mean_preproc_latency_ms: float
    mean_infer_latency_ms: float
    mean_tracking_latency_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_detections_per_frame: float
    mean_confidence: float
    map50: float
    map50_per_condition: Dict[str, float]
    count_eval: Dict[str, Any]
    timing_bucket_counts: Dict[str, int]
    preprocessing: Dict[str, Any]
    per_condition: Dict[str, Dict[str, float]]
    per_timing_bucket: Dict[str, Dict[str, float]]
    power: Dict[str, Any]
    tracking: Dict[str, Any]


class ConditionGatedEnhancers:
    """P3 enhancer interface. In day/night mode this uses Zero-DCE on night."""

    def __init__(
        self,
        zero_dce_module: Optional[str],
        derain_module: Optional[str],
        zero_dce_weights: Optional[str],
        derain_weights: Optional[str],
        enhancer_device: Optional[str],
        strict: bool,
    ) -> None:
        self.strict = strict
        self._warned_night = False
        self._warned_rain = False
        self.zero_dce = self._load_enhancer(
            module_name=zero_dce_module,
            fn_name="enhance",
            model_path=zero_dce_weights,
            device=enhancer_device,
            label="Zero-DCE",
        )
        self.derain = None
        if "rain" in VISUAL_CONDITIONS:
            self.derain = self._load_enhancer(
                module_name=derain_module,
                fn_name="enhance",
                model_path=derain_weights,
                device=enhancer_device,
                label="MobileDeRainGAN",
            )

    def _load_enhancer(
        self,
        module_name: Optional[str],
        fn_name: str,
        model_path: Optional[str],
        device: Optional[str],
        label: str,
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        if not module_name:
            return None

        module = importlib.import_module(module_name)
        cfg_fn = getattr(module, "configure", None)
        if callable(cfg_fn):
            cfg_fn(model_path=model_path, device=device, strict=self.strict)
        elif model_path:
            print(f"[WARN] {label} module '{module_name}' has no configure(); ignoring model path: {model_path}")

        fn = getattr(module, fn_name, None)
        if fn is None or not callable(fn):
            raise AttributeError(
                f"Module '{module_name}' must define callable '{fn_name}(frame: np.ndarray) -> np.ndarray'"
            )
        return fn

    def enhance_for_condition(self, frame: np.ndarray, condition: str) -> np.ndarray:
        if condition in {"night", "dusk", "low_light"}:
            if self.zero_dce is not None:
                return self.zero_dce(frame)
            if self.strict:
                raise RuntimeError("P3 requires Zero-DCE enhancer module/weights in strict mode.")
            if not self._warned_night:
                print("[WARN] Zero-DCE module missing; using identity transform for night in non-strict mode.")
                self._warned_night = True
            return frame

        if condition == "rain":
            if self.derain is not None:
                return self.derain(frame)
            if self.strict:
                raise RuntimeError("P3 requires MobileDeRainGAN enhancer module/weights in strict mode.")
            if not self._warned_rain:
                print("[WARN] MobileDeRainGAN module missing; using identity transform for rain in non-strict mode.")
                self._warned_rain = True
            return frame

        return frame


class IEEEExperiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.video_path = Path(args.video).expanduser().resolve()
        self.condition_videos = self._resolve_condition_videos()
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core_runs = build_core_runs(args.nmsfree_detector)
        self.lowlight_conditions: Set[str] = set(args.p3_lowlight_conditions or {"night"})
        self.gt_counts = self._load_gt_counts(args.gt_count)
        self.models: Dict[str, YOLO] = {}
        self.model_paths: Dict[str, Path] = {}
        self.roi_masks: Dict[str, np.ndarray] = {}
        self._load_roi_masks()
        self.enhancers = ConditionGatedEnhancers(
            zero_dce_module=args.zero_dce_module,
            derain_module=args.derain_module,
            zero_dce_weights=args.zero_dce_weights,
            derain_weights=args.derain_weights,
            enhancer_device=args.enhancer_device,
            strict=args.strict_p3,
        )
        if args.video_mode == "single":
            segment_path = Path(args.segments).resolve() if args.segments else None
            if segment_path is not None and not segment_path.exists():
                if args.strict_segments:
                    raise FileNotFoundError(f"Segment file not found: {segment_path}")
                print(f"[WARN] Segment file not found: {segment_path}. Falling back to auto thirds.")
                segment_path = None
            self.segments = self._load_segments(segment_path)
        else:
            self.segments = []
        self._validate_run_matrix()

    def _load_roi_masks(self) -> None:
        raw_paths = {
            "day": self.args.roi_mask_day,
            "dusk": self.args.roi_mask_dusk,
            "night": self.args.roi_mask_night,
            "mixed": self.args.roi_mask,
        }
        for cond, path_str in raw_paths.items():
            if not path_str:
                continue
            p = Path(path_str).expanduser().resolve()
            if not p.exists():
                print(f"[WARN] ROI mask for {cond} not found: {p}")
                continue
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self.roi_masks[cond] = mask
                print(f"[INFO] Loaded {cond} ROI mask: {p}")

    def _resolve_condition_videos(self) -> Dict[str, Path]:
        if self.args.video_mode == "single":
            if not self.video_path.exists():
                raise FileNotFoundError(f"Video not found: {self.video_path}")
            return {"mixed": self.video_path}

        raw_paths = {
            "day": self.args.video_day,
            "dusk": self.args.video_dusk,
            "night": self.args.video_night,
        }
        resolved: Dict[str, Path] = {}
        for cond in VISUAL_CONDITIONS:
            path = Path(raw_paths[cond]).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"Video for condition '{cond}' not found: {path}. "
                    "Pass --video-day/--video-dusk/--video-night explicitly."
                )
            resolved[cond] = path
        return resolved

    def _resolve_detector_path(self, detector_name: str) -> Path:
        candidates = DETECTOR_PATHS.get(detector_name)
        if not candidates:
            raise KeyError(f"Unknown detector '{detector_name}'. Known detectors: {sorted(DETECTOR_PATHS)}")
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No detector artifact found for '{detector_name}'. Checked: {candidates}"
            )
        return model_path

    def _validate_run_matrix(self) -> None:
        missing: Dict[str, str] = {}
        for run_cfg in self.core_runs:
            detector_name = run_cfg["detector"]
            if detector_name in missing:
                continue
            try:
                self._resolve_detector_path(detector_name)
            except Exception as exc:
                missing[detector_name] = str(exc)
        if missing:
            details = "\n".join(f"- {det}: {msg}" for det, msg in sorted(missing.items()))
            raise FileNotFoundError(
                "Experiment matrix references detector artifacts that are not available:\n" + details
            )

    def _load_segments(self, segment_file: Optional[Path]) -> List[Segment]:
        if segment_file is None:
            if self.args.strict_segments:
                raise FileNotFoundError(
                    "Segment file is required in strict mode. Provide --segments with day/dusk/night intervals."
                )
            print("[WARN] --segments not provided. Falling back to equal thirds of video duration.")
            return self._fallback_segments_by_thirds()

        raw = json.loads(segment_file.read_text())
        if not isinstance(raw, list):
            raise ValueError("Segment file must be a JSON list of {condition,start_sec,end_sec} entries.")

        segments: List[Segment] = []
        for row in raw:
            condition = str(row["condition"]).strip().lower()
            if condition not in VISUAL_CONDITIONS:
                raise ValueError(f"Invalid condition '{condition}'. Must be one of {VISUAL_CONDITIONS}.")
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if end_sec <= start_sec:
                raise ValueError(f"Invalid segment interval for {condition}: end_sec must be > start_sec.")
            segments.append(Segment(condition=condition, start_sec=start_sec, end_sec=end_sec))

        segments.sort(key=lambda s: s.start_sec)
        return segments

    def _fallback_segments_by_thirds(self) -> List[Segment]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration_sec = (frames / fps) if fps > 0 else 0.0
        one_third = duration_sec / 3.0 if duration_sec > 0 else 0.0
        two_thirds = 2.0 * one_third
        return [
            Segment("day", 0.0, one_third),
            Segment("dusk", one_third, two_thirds),
            Segment("night", two_thirds, max(duration_sec, two_thirds + 1.0)),
        ]

    def _condition_for_pts_ms(self, pts_ms: float) -> str:
        sec = pts_ms / 1000.0
        for seg in self.segments:
            if seg.start_sec <= sec < seg.end_sec:
                return seg.condition
        return self.segments[-1].condition

    def _load_model(self, detector_name: str) -> YOLO:
        if detector_name in self.models:
            return self.models[detector_name]

        model_path = self._resolve_detector_path(detector_name)

        print(f"Loading detector '{detector_name}' from {model_path}...")
        model = YOLO(str(model_path))
        self.models[detector_name] = model
        self.model_paths[detector_name] = model_path
        return model

    @staticmethod
    def _mean_luminance(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray) / 255.0)

    def _p3_gate_enabled(self, condition: str, mean_luma: float) -> bool:
        condition_gate = condition in self.lowlight_conditions
        luminance_gate = mean_luma <= float(self.args.tau_night)

        if self.args.p3_gating == "segment":
            return condition_gate
        if self.args.p3_gating == "luminance":
            return luminance_gate
        if self.args.p3_gating == "segment_and_luminance":
            return condition_gate and luminance_gate
        raise ValueError(f"Unsupported --p3-gating mode: {self.args.p3_gating}")

    def _apply_preproc(self, frame: np.ndarray, preproc: str, condition: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        mean_luma = self._mean_luminance(frame)
        if preproc == "P0":
            return frame, {"effective_mode": "P0", "enhanced": False, "mean_luminance": mean_luma}

        if preproc == "P1":
            # CLAHE on luminance channel.
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            return (
                cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR),
                {"effective_mode": "P1", "enhanced": True, "mean_luminance": mean_luma},
            )

        if preproc == "P3":
            use_enhancement = self._p3_gate_enabled(condition=condition, mean_luma=mean_luma)
            if use_enhancement:
                return (
                    self.enhancers.enhance_for_condition(frame, condition),
                    {"effective_mode": "P3", "enhanced": True, "mean_luminance": mean_luma},
                )
            return frame, {"effective_mode": "P0", "enhanced": False, "mean_luminance": mean_luma}

        raise ValueError(f"Unsupported preprocessing mode: {preproc}")

    def _load_gt_det(self, gt_path: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
        if not gt_path:
            return {}
        p = Path(gt_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GT detection file not found: {p}")

        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 6:
                continue
            frame_id = int(float(cols[0]))
            x = float(cols[2])
            y = float(cols[3])
            w = float(cols[4])
            h = float(cols[5])
            class_id = None
            if len(cols) >= 7:
                try:
                    class_id = int(float(cols[6]))
                except ValueError:
                    class_id = None

            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            gt_by_frame.setdefault(frame_id, []).append(
                {
                    "xyxy": xyxy,
                    "class_id": class_id,
                }
            )
        return gt_by_frame

    def _load_gt_counts(self, gt_count_path: Optional[str]) -> Dict[str, float]:
        if not gt_count_path:
            return {}
        p = Path(gt_count_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GT count file not found: {p}")

        counts: Dict[str, float] = {}
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("GT count file must include a header row.")
            norm_fields = {name.strip().lower(): name for name in reader.fieldnames}
            if "condition" not in norm_fields or "total_count" not in norm_fields:
                raise ValueError(
                    "GT count file requires columns: condition,total_count"
                )
            cond_key = norm_fields["condition"]
            count_key = norm_fields["total_count"]
            for row in reader:
                cond = str(row.get(cond_key, "")).strip().lower()
                if not cond:
                    continue
                if cond not in VISUAL_CONDITIONS:
                    raise ValueError(
                        f"Invalid condition '{cond}' in GT count file. Expected one of {VISUAL_CONDITIONS}."
                    )
                try:
                    total = float(str(row.get(count_key, "")).strip())
                except ValueError as exc:
                    raise ValueError(f"Invalid total_count for condition '{cond}' in {p}") from exc
                counts[cond] = total
        return counts

    def _calc_ap50(
        self,
        preds: List[Dict[str, Any]],
        gt_by_frame: Dict[int, List[Dict[str, Any]]],
        class_id: int,
    ) -> float:
        gt_cls: Dict[int, List[Dict[str, Any]]] = {}
        n_gt = 0
        for frame_id, boxes in gt_by_frame.items():
            items = []
            for gt in boxes:
                gt_cid = int(gt["class_id"]) if gt.get("class_id") is not None else -1
                if gt_cid != class_id:
                    continue
                items.append({"xyxy": gt["xyxy"], "matched": False})
            if items:
                gt_cls[frame_id] = items
                n_gt += len(items)

        if n_gt == 0:
            return 0.0

        pred_cls = [p for p in preds if int(p["class_id"]) == class_id]
        if not pred_cls:
            return 0.0
        pred_cls.sort(key=lambda p: float(p["score"]), reverse=True)

        tp = np.zeros(len(pred_cls), dtype=np.float32)
        fp = np.zeros(len(pred_cls), dtype=np.float32)

        for i, pred in enumerate(pred_cls):
            frame_id = int(pred["frame_id"])
            candidates = gt_cls.get(frame_id, [])
            best_j = -1
            best_iou = 0.0
            for j, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = self._bbox_iou(pred["xyxy"], gt["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= 0.5:
                tp[i] = 1.0
                candidates[best_j]["matched"] = True
            else:
                fp[i] = 1.0

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        recall = tp_c / float(n_gt)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    def _compute_map50(
        self,
        preds: List[Dict[str, Any]],
        gt_by_frame: Dict[int, List[Dict[str, Any]]],
        frame_ids: Optional[Set[int]] = None,
    ) -> float:
        if not gt_by_frame:
            return 0.0

        if frame_ids is not None:
            gt_subset = {k: v for k, v in gt_by_frame.items() if k in frame_ids}
            pred_subset = [p for p in preds if int(p["frame_id"]) in frame_ids]
        else:
            gt_subset = gt_by_frame
            pred_subset = preds

        if not gt_subset:
            return 0.0

        has_class_labels = any(
            box.get("class_id") is not None
            for boxes in gt_subset.values()
            for box in boxes
        )
        if has_class_labels:
            classes = sorted(
                {
                    int(box["class_id"])
                    for boxes in gt_subset.values()
                    for box in boxes
                    if box.get("class_id") is not None
                }
            )
            if self.args.vehicle_only_detection_eval:
                classes = [c for c in classes if c in VEHICLE_CLASS_IDS]
        else:
            classes = [-1]
            gt_subset = {
                frame_id: [{"xyxy": box["xyxy"], "class_id": -1} for box in boxes]
                for frame_id, boxes in gt_subset.items()
            }

        if not classes:
            return 0.0

        pred_for_eval: List[Dict[str, Any]] = []
        for p in pred_subset:
            cid = int(p["class_id"])
            if classes == [-1]:
                cid = -1
            pred_for_eval.append(
                {
                    "frame_id": int(p["frame_id"]),
                    "xyxy": p["xyxy"],
                    "score": float(p["score"]),
                    "class_id": cid,
                }
            )

        ap_list = [self._calc_ap50(pred_for_eval, gt_subset, c) for c in classes]
        if not ap_list:
            return 0.0
        return float(np.mean(np.asarray(ap_list, dtype=np.float64)))

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _summarize_power_csv(self, csv_path: Path) -> Dict[str, float]:
        if not csv_path.exists():
            return {}

        rows = 0
        power_vals: List[float] = []
        gpu_vals: List[float] = []
        cpu_vals: List[float] = []
        ram_vals: List[float] = []

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                p = self._safe_float(row.get("power_total_mw"))
                g = self._safe_float(row.get("gpu_load_pct"))
                c = self._safe_float(row.get("cpu_load_pct"))
                r = self._safe_float(row.get("ram_used_mb"))
                if p is not None:
                    power_vals.append(p)
                if g is not None:
                    gpu_vals.append(g)
                if c is not None:
                    cpu_vals.append(c)
                if r is not None:
                    ram_vals.append(r)

        def _mean(vals: List[float]) -> float:
            return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0

        return {
            "samples": float(rows),
            "avg_power_mw": _mean(power_vals),
            "avg_gpu_load_pct": _mean(gpu_vals),
            "avg_cpu_load_pct": _mean(cpu_vals),
            "avg_ram_used_mb": _mean(ram_vals),
        }

    def _classify_timing_buckets(self, dt_ms: np.ndarray, median_dt_ms: float) -> List[str]:
        buckets: List[str] = []
        stable_th = 1.25 * median_dt_ms
        stutter_th = 2.5 * median_dt_ms
        for dt in dt_ms:
            if dt <= 0:
                buckets.append("stable")
            elif dt <= stable_th:
                buckets.append("stable")
            elif dt <= stutter_th:
                buckets.append("stutter")
            else:
                buckets.append("drop")
        return buckets

    def _set_power_mode_if_requested(self, power_mode: str) -> None:
        if not self.args.apply_nvpmodel:
            return

        mapping = {
            "MAXN": self.args.nvpmodel_maxn,
            "50W": self.args.nvpmodel_50w,
            "30W": self.args.nvpmodel_30w,
        }
        mode_id = mapping[power_mode]
        if mode_id is None:
            raise ValueError(
                f"apply_nvpmodel is enabled but no mode id provided for {power_mode}. "
                "Pass --nvpmodel-maxn/--nvpmodel-50w/--nvpmodel-30w."
            )

        cmd = ["sudo", "nvpmodel", "-m", str(mode_id)]
        print(f"Switching power mode {power_mode}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0

    def _load_gt_mot(self, gt_path: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
        if not gt_path:
            return {}
        p = Path(gt_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"GT MOT file not found: {p}")
        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 6:
                continue
            frame_id = int(float(cols[0]))
            track_id = int(float(cols[1]))
            x = float(cols[2])
            y = float(cols[3])
            w = float(cols[4])
            h = float(cols[5])
            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            gt_by_frame.setdefault(frame_id, []).append({"id": track_id, "xyxy": xyxy})
        return gt_by_frame

    def _load_pred_tracks_mot(self, tracks_path: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
        if not tracks_path:
            return {}
        p = Path(tracks_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"External tracks file not found: {p}")
        tracks_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 6:
                continue
            frame_id = int(float(cols[0]))
            track_id = int(float(cols[1]))
            x = float(cols[2])
            y = float(cols[3])
            w = float(cols[4])
            h = float(cols[5])
            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            tracks_by_frame.setdefault(frame_id, []).append({"id": track_id, "xyxy": xyxy})
        return tracks_by_frame

    def _external_tracks_path(self, run_id: str, power_mode: str, condition: str) -> Optional[str]:
        template = self.args.external_tracks_template
        if not template:
            return None
        return template.format(
            run_id=run_id,
            power_mode=power_mode.lower(),
            POWER_MODE=power_mode,
            condition=condition,
            CONDITION=condition.upper(),
        )

    def _build_tracker(self):
        if not self.args.enable_tracking:
            return None
        if self.args.tracker == "bytetrack":
            return sv.ByteTrack(
                track_activation_threshold=self.args.track_activation_threshold,
                minimum_matching_threshold=self.args.track_matching_threshold,
                frame_rate=self.args.tracker_frame_rate,
            )
        if self.args.tracker == "nvdcf":
            # NvDCF runs in DeepStream; this runner consumes pre-exported MOT tracks.
            return None
        raise ValueError(f"Unknown tracker: {self.args.tracker}")

    def _summarize_tracking(
        self,
        tracks_by_frame: Dict[int, List[Dict[str, Any]]],
        frame_to_bucket: Dict[int, str],
        gt_by_frame: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not self.args.enable_tracking:
            return {"enabled": False}

        result: Dict[str, Any] = {
            "enabled": True,
            "tracker": self.args.tracker,
            "iou_match_threshold": float(self.args.tracking_iou_threshold),
            "gt_file": str(self.args.gt_mot) if self.args.gt_mot else None,
            "uses_ground_truth": bool(gt_by_frame),
            "global": {},
            "per_timing_bucket": {},
            "notes": [],
        }

        bucket_names = ["stable", "stutter", "drop"]
        for bucket in bucket_names:
            frame_ids = [f for f, b in frame_to_bucket.items() if b == bucket]
            result["per_timing_bucket"][bucket] = {
                "frames": int(len(frame_ids)),
                "tracked_boxes": int(sum(len(tracks_by_frame.get(f, [])) for f in frame_ids)),
                "id_switches": 0,
            }

        if not tracks_by_frame:
            result["notes"].append("No tracks produced by tracker.")
            return result

        # With MOT GT: compute IDSW with frame-wise IoU matching.
        if gt_by_frame:
            iou_thr = float(self.args.tracking_iou_threshold)
            last_pred_for_gt: Dict[int, int] = {}
            total_tp = total_fp = total_fn = 0
            idsw_total = 0
            idsw_by_bucket = {b: 0 for b in bucket_names}

            all_frames = sorted(set(frame_to_bucket.keys()) | set(gt_by_frame.keys()) | set(tracks_by_frame.keys()))
            for frame_id in all_frames:
                preds = tracks_by_frame.get(frame_id, [])
                gts = gt_by_frame.get(frame_id, [])
                bucket = frame_to_bucket.get(frame_id, "stable")

                unmatched_pred = set(range(len(preds)))
                unmatched_gt = set(range(len(gts)))
                matches: List[Tuple[int, int]] = []

                # Greedy IoU matching.
                pairs: List[Tuple[float, int, int]] = []
                for gi, gt in enumerate(gts):
                    for pi, pr in enumerate(preds):
                        iou = self._bbox_iou(gt["xyxy"], pr["xyxy"])
                        if iou >= iou_thr:
                            pairs.append((iou, gi, pi))
                pairs.sort(reverse=True, key=lambda x: x[0])
                for _, gi, pi in pairs:
                    if gi in unmatched_gt and pi in unmatched_pred:
                        unmatched_gt.remove(gi)
                        unmatched_pred.remove(pi)
                        matches.append((gi, pi))

                tp = len(matches)
                fp = len(unmatched_pred)
                fn = len(unmatched_gt)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                for gi, pi in matches:
                    gt_id = int(gts[gi]["id"])
                    pred_id = int(preds[pi]["id"])
                    if gt_id in last_pred_for_gt:
                        if last_pred_for_gt[gt_id] != pred_id:
                            idsw_total += 1
                            if bucket in idsw_by_bucket:
                                idsw_by_bucket[bucket] += 1
                    last_pred_for_gt[gt_id] = pred_id

            result["global"] = {
                "id_switches": int(idsw_total),
                "matches_tp": int(total_tp),
                "false_positives": int(total_fp),
                "false_negatives": int(total_fn),
            }
            for bucket in bucket_names:
                result["per_timing_bucket"][bucket]["id_switches"] = int(idsw_by_bucket[bucket])
            return result

        # Without GT: report ID-churn proxy only.
        idsw_proxy_total = 0
        idsw_proxy_by_bucket = {b: 0 for b in bucket_names}
        frame_ids_sorted = sorted(frame_to_bucket.keys())
        for prev_frame, cur_frame in zip(frame_ids_sorted[:-1], frame_ids_sorted[1:]):
            prev_tracks = tracks_by_frame.get(prev_frame, [])
            cur_tracks = tracks_by_frame.get(cur_frame, [])
            bucket = frame_to_bucket.get(cur_frame, "stable")
            used = set()
            for p in prev_tracks:
                best_iou = 0.0
                best_j = -1
                for j, c in enumerate(cur_tracks):
                    if j in used:
                        continue
                    iou = self._bbox_iou(p["xyxy"], c["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0 and best_iou >= float(self.args.tracking_iou_threshold):
                    used.add(best_j)
                    if int(p["id"]) != int(cur_tracks[best_j]["id"]):
                        idsw_proxy_total += 1
                        if bucket in idsw_proxy_by_bucket:
                            idsw_proxy_by_bucket[bucket] += 1

        result["global"] = {
            "id_switches_proxy": int(idsw_proxy_total),
        }
        for bucket in bucket_names:
            result["per_timing_bucket"][bucket]["id_switches"] = int(idsw_proxy_by_bucket[bucket])
        result["notes"].append(
            "No GT provided: reported id_switches_proxy only. Provide --gt-mot for IDSW."
        )
        return result

    def run_single(
        self,
        run_cfg: Dict[str, str],
        power_mode: str,
        video_path: Optional[Path] = None,
        condition_override: Optional[str] = None,
    ) -> RunSummary:
        run_id = run_cfg["run_id"]
        detector_name = run_cfg["detector"]
        preproc = run_cfg["preproc"]
        effective_video = (video_path or self.video_path).expanduser().resolve()
        condition_label = condition_override or "mixed"

        model = self._load_model(detector_name)
        self._set_power_mode_if_requested(power_mode)

        cap = cv2.VideoCapture(str(effective_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {effective_video}")

        power_proc = None
        run_suffix = f"_{condition_label}" if condition_label != "mixed" else ""
        power_csv = self.output_dir / f"{run_id}_{power_mode}{run_suffix}_power.csv"
        if self.args.with_power:
            power_cmd = [
                sys.executable,
                str(REPO_ROOT / "monitor_power.py"),
                "--run-id",
                f"{run_id}_{power_mode}{run_suffix}",
                "--output",
                str(power_csv),
                "--interval",
                str(self.args.power_interval),
            ]
            power_proc = subprocess.Popen(power_cmd)
            time.sleep(float(self.args.power_warmup_sec))

        records: List[FrameRecord] = []
        tracker = self._build_tracker()
        gt_by_frame = self._load_gt_mot(self.args.gt_mot)
        gt_det_by_frame = self._load_gt_det(self.args.gt_det or self.args.gt_mot)
        tracks_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        external_tracks_path = None
        if self.args.enable_tracking and self.args.tracker == "nvdcf":
            external_tracks_path = self._external_tracks_path(run_id, power_mode, condition_label)
            tracks_by_frame = self._load_pred_tracks_mot(external_tracks_path)
        preds_for_map: List[Dict[str, Any]] = []
        prev_pts = 0.0
        frame_idx = 0

        p3_prev_effective_mode: Optional[str] = None
        p3_switches = 0
        p3_switch_event_pre_ms: List[float] = []
        p3_steady_pre_ms: List[float] = []
        p3_enhanced_frames = 0
        p3_raw_frames = 0
        p3_luminance: List[float] = []

        max_frames = self.args.max_frames
        budget_ms = SIMULATED_POWER_BUDGET_MS[power_mode]

        print(
            f"\n[{run_id} | {power_mode}{run_suffix}] detector={detector_name}, "
            f"preproc={preproc}, video={effective_video.name}"
        )
        wall_start = time.time()

        try:
            while True:
                if max_frames > 0 and frame_idx >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                pts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                dt_ms = pts_ms - prev_pts if frame_idx > 0 else 0.0
                prev_pts = pts_ms

                condition = condition_override if condition_override is not None else self._condition_for_pts_ms(pts_ms)

                t_pre = time.time()
                proc, pre_meta = self._apply_preproc(frame, preproc, condition)
                preproc_ms = (time.time() - t_pre) * 1000.0

                t_inf = time.time()
                results = model(proc, verbose=False)
                infer_ms = (time.time() - t_inf) * 1000.0

                boxes = results[0].boxes
                num_det = int(len(boxes))
                mean_conf = float(boxes.conf.mean().item()) if num_det > 0 else 0.0

                if num_det > 0:
                    xyxy_all = boxes.xyxy.detach().cpu().numpy()
                    conf_all = boxes.conf.detach().cpu().numpy()
                    cls_all = boxes.cls.detach().cpu().numpy().astype(np.int32)
                else:
                    xyxy_all = np.empty((0, 4), dtype=np.float32)
                    conf_all = np.empty((0,), dtype=np.float32)
                    cls_all = np.empty((0,), dtype=np.int32)

                if gt_det_by_frame:
                    for i in range(len(xyxy_all)):
                        cid = int(cls_all[i]) if len(cls_all) > i else -1
                        if self.args.vehicle_only_detection_eval and cid not in VEHICLE_CLASS_IDS:
                            continue
                        preds_for_map.append(
                            {
                                "frame_id": int(frame_idx + 1),
                                "xyxy": np.asarray(xyxy_all[i], dtype=np.float32),
                                "score": float(conf_all[i]) if len(conf_all) > i else 0.0,
                                "class_id": cid,
                            }
                        )

                tracking_ms = 0.0
                if tracker is not None:
                    xyxy = xyxy_all
                    conf_arr = conf_all
                    cls_arr = cls_all
                    if self.args.vehicle_only_tracking and len(cls_arr) > 0:
                        mask = np.isin(cls_arr, np.array(sorted(VEHICLE_CLASS_IDS), dtype=np.int32))
                        xyxy = xyxy[mask]
                        conf_arr = conf_arr[mask]
                        cls_arr = cls_arr[mask]

                    # ByteTrack expects ndarray fields even when there are zero detections.
                    det_sv = sv.Detections(
                        xyxy=np.asarray(xyxy, dtype=np.float32),
                        confidence=np.asarray(conf_arr, dtype=np.float32),
                        class_id=np.asarray(cls_arr, dtype=np.int32),
                    )

                    # ROI filtering: only keep detections whose center is inside the ROI mask
                    current_mask = self.roi_masks.get(condition)
                    if current_mask is not None and len(det_sv) > 0:
                        mask_h, mask_w = current_mask.shape[:2]
                        frame_h, frame_w = proc.shape[:2]
                        if mask_h != frame_h or mask_w != frame_w:
                            # Cache resized mask? For now, resize per change or per frame
                            current_mask = cv2.resize(current_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                            self.roi_masks[condition] = current_mask

                        centers = det_sv.get_anchors_coordinates(sv.Position.CENTER).astype(int)
                        roi_keep = []
                        for cx, cy in centers:
                            if 0 <= cx < frame_w and 0 <= cy < frame_h:
                                roi_keep.append(current_mask[cy, cx] > 0)
                            else:
                                roi_keep.append(False)
                        det_sv = det_sv[np.array(roi_keep, dtype=bool)]

                    t_track = time.time()
                    tracked_sv = tracker.update_with_detections(det_sv)
                    tracking_ms = (time.time() - t_track) * 1000.0
                    frame_id = frame_idx + 1  # MOT convention
                    frame_tracks: List[Dict[str, Any]] = []
                    if tracked_sv.tracker_id is not None and len(tracked_sv.xyxy) > 0:
                        for i in range(len(tracked_sv.xyxy)):
                            tid = tracked_sv.tracker_id[i]
                            if tid is None:
                                continue
                            if isinstance(tid, float) and np.isnan(tid):
                                continue
                            frame_tracks.append(
                                {
                                    "id": int(tid),
                                    "xyxy": np.asarray(tracked_sv.xyxy[i], dtype=np.float32),
                                }
                            )
                    tracks_by_frame[frame_id] = frame_tracks

                total_latency_ms = preproc_ms + infer_ms + tracking_ms

                if preproc == "P3":
                    p3_effective_mode = str(pre_meta.get("effective_mode", "P0"))
                    p3_is_enhanced = bool(pre_meta.get("enhanced", False))
                    mean_luma = float(pre_meta.get("mean_luminance", 0.0))
                    p3_luminance.append(mean_luma)
                    if p3_is_enhanced:
                        p3_enhanced_frames += 1
                    else:
                        p3_raw_frames += 1
                    if p3_prev_effective_mode is None:
                        p3_steady_pre_ms.append(preproc_ms)
                    elif p3_prev_effective_mode != p3_effective_mode:
                        p3_switches += 1
                        p3_switch_event_pre_ms.append(preproc_ms)
                    else:
                        p3_steady_pre_ms.append(preproc_ms)
                    p3_prev_effective_mode = p3_effective_mode
                else:
                    p3_is_enhanced = False
                    mean_luma = float(pre_meta.get("mean_luminance", 0.0))

                records.append(
                    FrameRecord(
                        frame_idx=frame_idx,
                        pts_ms=float(pts_ms),
                        dt_ms=float(dt_ms),
                        visual_condition=condition,
                        timing_bucket="unknown",
                        preproc_latency_ms=float(preproc_ms),
                        infer_latency_ms=float(infer_ms),
                        tracking_latency_ms=float(tracking_ms),
                        latency_ms=float(total_latency_ms),
                        num_detections=num_det,
                        mean_confidence=mean_conf,
                        p3_enhanced=p3_is_enhanced,
                        mean_luminance=mean_luma,
                    )
                )

                if self.args.simulate_power_budget and total_latency_ms < budget_ms:
                    time.sleep((budget_ms - total_latency_ms) / 1000.0)

                frame_idx += 1
        finally:
            cap.release()
            if power_proc is not None:
                power_proc.terminate()
                try:
                    power_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    power_proc.kill()
        wall_end = time.time()
        wall_time_sec = max(wall_end - wall_start, 1e-9)
        fps = float(len(records) / wall_time_sec) if records else 0.0

        if not records:
            raise RuntimeError(f"No frames processed for {run_id} in power mode {power_mode}.")

        dt_values = np.array([r.dt_ms for r in records if r.dt_ms > 0], dtype=np.float64)
        median_dt_ms = float(np.median(dt_values)) if len(dt_values) > 0 else 33.33

        dt_all = np.array([r.dt_ms for r in records], dtype=np.float64)
        buckets = self._classify_timing_buckets(dt_all, median_dt_ms)
        for r, b in zip(records, buckets):
            r.timing_bucket = b
        frame_to_bucket = {int(r.frame_idx + 1): r.timing_bucket for r in records}
        tracking_summary = self._summarize_tracking(
            tracks_by_frame=tracks_by_frame,
            frame_to_bucket=frame_to_bucket,
            gt_by_frame=gt_by_frame,
        )
        if self.args.enable_tracking and self.args.tracker == "nvdcf":
            tracking_summary.setdefault("notes", []).append(
                f"Consumed external NvDCF tracks: {external_tracks_path}"
            )

        unique_track_ids = (
            {
                int(item["id"])
                for frame_items in tracks_by_frame.values()
                for item in frame_items
                if item.get("id") is not None
            }
            if self.args.enable_tracking
            else set()
        )
        pred_total_count = float(len(unique_track_ids)) if unique_track_ids else None
        gt_total_count = None
        if self.gt_counts:
            if condition_label in self.gt_counts:
                gt_total_count = float(self.gt_counts[condition_label])
            elif condition_label == "mixed":
                gt_total_count = float(sum(self.gt_counts.get(c, 0.0) for c in VISUAL_CONDITIONS))
        abs_error = None
        rel_error_pct = None
        if gt_total_count is not None and pred_total_count is not None:
            abs_error = abs(pred_total_count - gt_total_count)
            rel_error_pct = (abs_error / gt_total_count * 100.0) if gt_total_count > 0 else 0.0
        count_eval = {
            "gt_total_count": gt_total_count,
            "pred_total_count": pred_total_count,
            "abs_error": abs_error,
            "relative_error_pct": rel_error_pct,
            "requires_tracking_for_pred": True,
            "tracking_enabled": bool(self.args.enable_tracking),
        }

        map50 = self._compute_map50(preds_for_map, gt_det_by_frame)
        map50_per_condition: Dict[str, float] = {}
        for cond in VISUAL_CONDITIONS:
            cond_frames = {int(r.frame_idx + 1) for r in records if r.visual_condition == cond}
            map50_per_condition[cond] = self._compute_map50(
                preds_for_map,
                gt_det_by_frame,
                frame_ids=cond_frames,
            )

        power_stats = self._summarize_power_csv(power_csv) if self.args.with_power else {}
        avg_power_mw = float(power_stats.get("avg_power_mw", 0.0))
        fps_per_watt = float(fps / (avg_power_mw / 1000.0)) if avg_power_mw > 0 else 0.0

        p3_switch_event_mean_ms = (
            float(np.mean(np.asarray(p3_switch_event_pre_ms, dtype=np.float64)))
            if p3_switch_event_pre_ms
            else 0.0
        )
        p3_steady_mean_ms = (
            float(np.mean(np.asarray(p3_steady_pre_ms, dtype=np.float64)))
            if p3_steady_pre_ms
            else 0.0
        )
        p3_switch_overhead_ms = max(0.0, p3_switch_event_mean_ms - p3_steady_mean_ms)

        preproc_meta = {
            "mode": preproc,
            "p3_gating": self.args.p3_gating if preproc == "P3" else "",
            "tau_night": float(self.args.tau_night) if preproc == "P3" else 0.0,
            "p3_lowlight_conditions": sorted(self.lowlight_conditions) if preproc == "P3" else [],
            "p3_switch_count": int(p3_switches) if preproc == "P3" else 0,
            "p3_enhanced_frames": int(p3_enhanced_frames) if preproc == "P3" else 0,
            "p3_passthrough_frames": int(p3_raw_frames) if preproc == "P3" else 0,
            "p3_switch_event_pre_mean_ms": p3_switch_event_mean_ms if preproc == "P3" else 0.0,
            "p3_steady_pre_mean_ms": p3_steady_mean_ms if preproc == "P3" else 0.0,
            "p3_switch_overhead_ms": p3_switch_overhead_ms if preproc == "P3" else 0.0,
            "mean_luminance": (
                float(np.mean(np.asarray(p3_luminance, dtype=np.float64)))
                if p3_luminance
                else 0.0
            ),
        }

        lat = np.array([r.latency_ms for r in records], dtype=np.float64)
        lat_pre = np.array([r.preproc_latency_ms for r in records], dtype=np.float64)
        lat_inf = np.array([r.infer_latency_ms for r in records], dtype=np.float64)
        lat_track = np.array([r.tracking_latency_ms for r in records], dtype=np.float64)
        det = np.array([r.num_detections for r in records], dtype=np.float64)
        conf = np.array([r.mean_confidence for r in records], dtype=np.float64)

        per_condition: Dict[str, Dict[str, float]] = {}
        for c in VISUAL_CONDITIONS:
            idx = [i for i, r in enumerate(records) if r.visual_condition == c]
            if not idx:
                per_condition[c] = {
                    "frames": 0.0,
                    "mean_preproc_latency_ms": 0.0,
                    "mean_infer_latency_ms": 0.0,
                    "mean_tracking_latency_ms": 0.0,
                    "mean_latency_ms": 0.0,
                    "mean_detections": 0.0,
                    "mean_confidence": 0.0,
                    "map50": map50_per_condition.get(c, 0.0),
                }
                continue
            per_condition[c] = {
                "frames": float(len(idx)),
                "mean_preproc_latency_ms": float(lat_pre[idx].mean()),
                "mean_infer_latency_ms": float(lat_inf[idx].mean()),
                "mean_tracking_latency_ms": float(lat_track[idx].mean()),
                "mean_latency_ms": float(lat[idx].mean()),
                "mean_detections": float(det[idx].mean()),
                "mean_confidence": float(conf[idx].mean()),
                "map50": map50_per_condition.get(c, 0.0),
            }

        per_timing: Dict[str, Dict[str, float]] = {}
        for b in ["stable", "stutter", "drop"]:
            idx = [i for i, r in enumerate(records) if r.timing_bucket == b]
            if not idx:
                per_timing[b] = {
                    "frames": 0.0,
                    "mean_preproc_latency_ms": 0.0,
                    "mean_infer_latency_ms": 0.0,
                    "mean_tracking_latency_ms": 0.0,
                    "mean_latency_ms": 0.0,
                    "mean_detections": 0.0,
                    "mean_confidence": 0.0,
                }
                continue
            per_timing[b] = {
                "frames": float(len(idx)),
                "mean_preproc_latency_ms": float(lat_pre[idx].mean()),
                "mean_infer_latency_ms": float(lat_inf[idx].mean()),
                "mean_tracking_latency_ms": float(lat_track[idx].mean()),
                "mean_latency_ms": float(lat[idx].mean()),
                "mean_detections": float(det[idx].mean()),
                "mean_confidence": float(conf[idx].mean()),
            }

        timing_counts = {
            "stable": int(sum(1 for r in records if r.timing_bucket == "stable")),
            "stutter": int(sum(1 for r in records if r.timing_bucket == "stutter")),
            "drop": int(sum(1 for r in records if r.timing_bucket == "drop")),
        }

        summary = RunSummary(
            run_id=run_id,
            video=str(effective_video),
            video_condition=condition_label,
            detector=detector_name,
            detector_path=str(self.model_paths.get(detector_name, DETECTOR_PATHS[detector_name][0])),
            preproc=preproc,
            power_mode=power_mode,
            frames_processed=int(len(records)),
            wall_time_sec=float(wall_time_sec),
            fps=float(fps),
            fps_per_watt=float(fps_per_watt),
            avg_power_mw=float(avg_power_mw),
            median_dt_ms=float(median_dt_ms),
            mean_preproc_latency_ms=float(lat_pre.mean()),
            mean_infer_latency_ms=float(lat_inf.mean()),
            mean_tracking_latency_ms=float(lat_track.mean()),
            mean_latency_ms=float(lat.mean()),
            p95_latency_ms=float(np.percentile(lat, 95)),
            mean_detections_per_frame=float(det.mean()),
            mean_confidence=float(conf.mean()),
            map50=float(map50),
            map50_per_condition=map50_per_condition,
            count_eval=count_eval,
            timing_bucket_counts=timing_counts,
            preprocessing=preproc_meta,
            per_condition=per_condition,
            per_timing_bucket=per_timing,
            power=power_stats,
            tracking=tracking_summary,
        )

        run_file = self.output_dir / f"{run_id}_{power_mode}{run_suffix}.json"
        run_file.write_text(json.dumps(asdict(summary), indent=2))
        print(f"  -> saved {run_file}")
        return summary

    def _aggregate_summaries(self, all_summaries: List[RunSummary]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for s in all_summaries:
            key = (s.run_id, s.detector, s.preproc, s.power_mode)
            if key not in grouped:
                grouped[key] = {
                    "run_id": s.run_id,
                    "detector": s.detector,
                    "preproc": s.preproc,
                    "power_mode": s.power_mode,
                    "frames_total": 0,
                    "wall_time_total_sec": 0.0,
                    "latency_weighted_sum": 0.0,
                    "power_time_sum_sec": 0.0,
                    "power_weighted_sum": 0.0,
                    "gt_count_sum": 0.0,
                    "pred_count_sum": 0.0,
                    "has_gt_count": True,
                    "has_pred_count": True,
                    "conditions": {},
                }

            g = grouped[key]
            frames = int(s.frames_processed)
            wall = float(s.wall_time_sec)
            g["frames_total"] += frames
            g["wall_time_total_sec"] += wall
            g["latency_weighted_sum"] += float(s.mean_latency_ms) * frames
            if float(s.avg_power_mw) > 0 and wall > 0:
                g["power_time_sum_sec"] += wall
                g["power_weighted_sum"] += float(s.avg_power_mw) * wall

            cond = str(s.video_condition)
            cond_map = float(s.map50_per_condition.get(cond, s.map50))
            gt_count = s.count_eval.get("gt_total_count")
            pred_count = s.count_eval.get("pred_total_count")
            if gt_count is None:
                g["has_gt_count"] = False
            else:
                g["gt_count_sum"] += float(gt_count)
            if pred_count is None:
                g["has_pred_count"] = False
            else:
                g["pred_count_sum"] += float(pred_count)
            g["conditions"][cond] = {
                "video": s.video,
                "frames": frames,
                "latency_ms": float(s.mean_latency_ms),
                "map50": cond_map,
                "gt_total_count": gt_count,
                "pred_total_count": pred_count,
                "count_abs_error": s.count_eval.get("abs_error"),
                "count_relative_error_pct": s.count_eval.get("relative_error_pct"),
            }

        rows: List[Dict[str, Any]] = []
        for key in sorted(grouped.keys()):
            g = grouped[key]
            frames_total = int(g["frames_total"])
            wall_total = float(g["wall_time_total_sec"])
            avg_latency = float(g["latency_weighted_sum"] / max(frames_total, 1))
            fps = float(frames_total / max(wall_total, 1e-9))

            if g["power_time_sum_sec"] > 0:
                avg_power_mw = float(g["power_weighted_sum"] / g["power_time_sum_sec"])
            else:
                avg_power_mw = 0.0
            fps_per_watt = float(fps / (avg_power_mw / 1000.0)) if avg_power_mw > 0 else 0.0

            day_map = float(g["conditions"].get("day", {}).get("map50", 0.0))
            dusk_map = float(g["conditions"].get("dusk", {}).get("map50", 0.0))
            night_map = float(g["conditions"].get("night", {}).get("map50", 0.0))
            present_maps = [
                float(v.get("map50", 0.0))
                for v in g["conditions"].values()
                if v.get("map50", None) is not None
            ]
            mean_map = float(np.mean(np.asarray(present_maps, dtype=np.float64))) if present_maps else 0.0
            gt_total_count = float(g["gt_count_sum"]) if g["has_gt_count"] else None
            pred_total_count = float(g["pred_count_sum"]) if g["has_pred_count"] else None
            count_abs_error = None
            count_relative_error_pct = None
            if gt_total_count is not None and pred_total_count is not None:
                count_abs_error = abs(pred_total_count - gt_total_count)
                count_relative_error_pct = (count_abs_error / gt_total_count * 100.0) if gt_total_count > 0 else 0.0

            rows.append(
                {
                    "run_id": g["run_id"],
                    "detector": g["detector"],
                    "preproc": g["preproc"],
                    "power_mode": g["power_mode"],
                    "frames_total": frames_total,
                    "latency_ms": avg_latency,
                    "fps": fps,
                    "avg_power_mw": avg_power_mw,
                    "fps_per_watt": fps_per_watt,
                    "day_map50": day_map,
                    "dusk_map50": dusk_map,
                    "night_map50": night_map,
                    "map50_mean": mean_map,
                    "gt_total_count": gt_total_count,
                    "pred_total_count": pred_total_count,
                    "count_abs_error": count_abs_error,
                    "count_relative_error_pct": count_relative_error_pct,
                    "conditions_present": sorted(g["conditions"].keys()),
                    "conditions": g["conditions"],
                }
            )
        return rows

    def run_all(self) -> None:
        if self.args.video_mode == "three":
            video_meta: Dict[str, Any] = {
                cond: str(path)
                for cond, path in sorted(self.condition_videos.items())
            }
            segments_meta: List[Dict[str, Any]] = []
        else:
            video_meta = str(self.video_path)
            segments_meta = [asdict(s) for s in self.segments]

        meta = {
            "video_mode": self.args.video_mode,
            "video": video_meta,
            "segments": segments_meta,
            "power_modes": POWER_MODES,
            "runs": self.core_runs,
            "nmsfree_detector": self.args.nmsfree_detector,
            "preprocessing": {
                "p3_gating": self.args.p3_gating,
                "tau_night": float(self.args.tau_night),
                "p3_lowlight_conditions": sorted(self.lowlight_conditions),
            },
            "detection_eval": {
                "gt_det": str(self.args.gt_det) if self.args.gt_det else None,
                "vehicle_only_detection_eval": bool(self.args.vehicle_only_detection_eval),
            },
            "count_eval": {
                "gt_count": str(self.args.gt_count) if self.args.gt_count else None,
                "method": "unique_track_ids_per_run_condition",
            },
            "power_logging": {
                "enabled": bool(self.args.with_power),
                "interval_sec": float(self.args.power_interval),
            },
            "tracking": {
                "enabled": bool(self.args.enable_tracking),
                "tracker": self.args.tracker,
                "gt_mot": str(self.args.gt_mot) if self.args.gt_mot else None,
                "external_tracks_template": (
                    str(self.args.external_tracks_template) if self.args.external_tracks_template else None
                ),
                "vehicle_only_tracking": bool(self.args.vehicle_only_tracking),
                "tracking_iou_threshold": float(self.args.tracking_iou_threshold),
            },
            "timestamp_buckets": {
                "stable": "dt <= 1.25 * median_dt",
                "stutter": "1.25 * median_dt < dt <= 2.5 * median_dt",
                "drop": "dt > 2.5 * median_dt",
            },
        }
        (self.output_dir / "protocol_meta.json").write_text(json.dumps(meta, indent=2))

        all_summaries: List[RunSummary] = []
        run_targets: List[Tuple[str, Path, Optional[str]]] = []
        if self.args.video_mode == "three":
            for cond in VISUAL_CONDITIONS:
                run_targets.append((cond, self.condition_videos[cond], cond))
        else:
            run_targets.append(("mixed", self.video_path, None))

        active_runs = self.core_runs
        if self.args.only_runs:
            only = set(self.args.only_runs)
            active_runs = [r for r in self.core_runs if r["run_id"] in only]

        total = len(POWER_MODES) * len(active_runs) * len(run_targets)
        step = 0

        for power_mode in POWER_MODES:
            for run_cfg in active_runs:
                for _, video_path, cond_override in run_targets:
                    step += 1
                    cond_lbl = cond_override if cond_override is not None else "mixed"
                    print(f"\n=== [{step}/{total}] {run_cfg['run_id']} @ {power_mode} ({cond_lbl}) ===")
                    summary = self.run_single(
                        run_cfg,
                        power_mode,
                        video_path=video_path,
                        condition_override=cond_override,
                    )
                    all_summaries.append(summary)

        aggregated_rows = self._aggregate_summaries(all_summaries)

        summary_payload = {
            "total_runs": len(all_summaries),
            "total_aggregated_runs": len(aggregated_rows),
            "video_mode": self.args.video_mode,
            "runs": [asdict(s) for s in all_summaries],
            "aggregated_runs": aggregated_rows,
        }
        summary_file = self.output_dir / "summary.json"
        summary_file.write_text(json.dumps(summary_payload, indent=2))
        agg_json_file = self.output_dir / "summary_aggregated.json"
        agg_json_file.write_text(json.dumps({"runs": aggregated_rows}, indent=2))

        # CSV-like markdown table aligned with the compact benchmark summary shape.
        md_lines = [
            "# Detection/Latency Summary (Table-I layout)",
            "",
            "| Config | Power | Video Cond | Pre-proc | Latency ms | FPS | FPS/W | Day mAP@0.5 | Dusk mAP@0.5 | Night mAP@0.5 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in all_summaries:
            md_lines.append(
                "| {run_id} ({det}) | {pwr} | {cond} | {pre} | {lat:.2f} | {fps:.2f} | {fpw:.3f} | {d:.3f} | {du:.3f} | {n:.3f} |".format(
                    run_id=s.run_id,
                    det=s.detector,
                    pwr=s.power_mode,
                    cond=s.video_condition,
                    pre=s.preproc,
                    lat=s.mean_latency_ms,
                    fps=s.fps,
                    fpw=s.fps_per_watt,
                    d=s.map50_per_condition.get("day", 0.0),
                    du=s.map50_per_condition.get("dusk", 0.0),
                    n=s.map50_per_condition.get("night", 0.0),
                )
            )

        table_file = self.output_dir / "table_detection_latency.md"
        table_file.write_text("\n".join(md_lines) + "\n")

        agg_lines = [
            "# Detection/Latency Summary (Aggregated by Run + Power)",
            "",
            "| Config | Power | Pre-proc | Latency ms | FPS | FPS/W | Day mAP@0.5 | Dusk mAP@0.5 | Night mAP@0.5 | Mean mAP@0.5 | GT Count | Pred Count | Abs Err | Rel Err % |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in aggregated_rows:
            agg_lines.append(
                "| {run_id} ({det}) | {pwr} | {pre} | {lat:.2f} | {fps:.2f} | {fpw:.3f} | {d:.3f} | {du:.3f} | {n:.3f} | {m:.3f} | {gt} | {pred} | {ae} | {re} |".format(
                    run_id=row["run_id"],
                    det=row["detector"],
                    pwr=row["power_mode"],
                    pre=row["preproc"],
                    lat=float(row["latency_ms"]),
                    fps=float(row["fps"]),
                    fpw=float(row["fps_per_watt"]),
                    d=float(row["day_map50"]),
                    du=float(row["dusk_map50"]),
                    n=float(row["night_map50"]),
                    m=float(row["map50_mean"]),
                    gt=(
                        f"{float(row['gt_total_count']):.0f}"
                        if row.get("gt_total_count") is not None
                        else "NA"
                    ),
                    pred=(
                        f"{float(row['pred_total_count']):.0f}"
                        if row.get("pred_total_count") is not None
                        else "NA"
                    ),
                    ae=(
                        f"{float(row['count_abs_error']):.0f}"
                        if row.get("count_abs_error") is not None
                        else "NA"
                    ),
                    re=(
                        f"{float(row['count_relative_error_pct']):.2f}"
                        if row.get("count_relative_error_pct") is not None
                        else "NA"
                    ),
                )
            )
        agg_table_file = self.output_dir / "table_detection_latency_aggregated.md"
        agg_table_file.write_text("\n".join(agg_lines) + "\n")

        print("\n========================================")
        print("IEEE protocol experiment complete")
        print(f"Summary: {summary_file}")
        print(f"Aggregated summary: {agg_json_file}")
        print(f"Table:   {table_file}")
        print(f"Aggregated table: {agg_table_file}")
        print("========================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traffic-counting benchmark experiment")
    parser.add_argument(
        "--video-mode",
        default="three",
        choices=["three", "single"],
        help=(
            "Video input mode: 'three' uses day/dusk/night files; "
            "'single' uses --video plus --segments."
        ),
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Path to source video stream")
    parser.add_argument("--video-day", default=str(DEFAULT_VIDEO_DAY), help="Day video path (three-video mode)")
    parser.add_argument("--video-dusk", default=str(DEFAULT_VIDEO_DUSK), help="Dusk video path (three-video mode)")
    parser.add_argument("--video-night", default=str(DEFAULT_VIDEO_NIGHT), help="Night video path (three-video mode)")
    parser.add_argument(
        "--segments",
        default=str(DEFAULT_SEGMENTS),
        help=(
            "JSON list file with day/dusk/night segments: "
            "[{\"condition\":\"day\",\"start_sec\":0,\"end_sec\":3600}, ...]"
        ),
    )
    parser.add_argument("--roi-mask", default="", help="ROI mask for single/mixed mode (white=count)")
    parser.add_argument("--roi-mask-day", default="", help="ROI mask for day segment")
    parser.add_argument("--roi-mask-dusk", default="", help="ROI mask for dusk segment")
    parser.add_argument("--roi-mask-night", default="", help="ROI mask for night segment")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=1000, help="Max frames per run (<=0 means full video)")
    parser.add_argument(
        "--nmsfree-detector",
        default="yolo11n",
        choices=["yolov10n", "yolo11n"],
        help="Detector used for the R2/R5 NMS-free comparison family.",
    )

    parser.add_argument(
        "--only-runs",
        nargs="+",
        metavar="RUN_ID",
        default=None,
        help="Only execute the listed run IDs (e.g. --only-runs R1 R3). Skips all others.",
    )
    parser.add_argument(
        "--strict-segments",
        action="store_true",
        help="Require explicit day/dusk/night segments file for strict reproducible runs",
    )
    parser.add_argument(
        "--strict-p3",
        action="store_true",
        help="Require Zero-DCE module and weights for P3 runs",
    )
    parser.add_argument(
        "--zero-dce-module",
        default="enhancers.zero_dce_wrapper",
        help="Python module for Zero-DCE wrapper exposing enhance(frame)",
    )
    parser.add_argument(
        "--derain-module",
        default="",
        help="Optional rain enhancer module exposing enhance(frame) (unused in day/night mode).",
    )
    parser.add_argument(
        "--zero-dce-weights",
        default="",
        help="TorchScript weights for Zero-DCE (default wrapper also checks ZERO_DCE_MODEL_PATH env var).",
    )
    parser.add_argument(
        "--derain-weights",
        default="",
        help="Optional rain-enhancer weights path (unused in day/night mode).",
    )
    parser.add_argument(
        "--enhancer-device",
        default="",
        help="Device for enhancer modules, e.g. cuda:0 or cpu (empty => wrapper default).",
    )
    parser.add_argument(
        "--p3-gating",
        default="segment_and_luminance",
        choices=["segment", "luminance", "segment_and_luminance"],
        help="Gating policy for P3 enhancement application.",
    )
    parser.add_argument(
        "--tau-night",
        type=float,
        default=0.35,
        help="Global mean luminance threshold for low-light gating (0..1).",
    )
    parser.add_argument(
        "--p3-lowlight-conditions",
        default="dusk,night",
        help="Comma-separated segment labels considered low-light for segment gating.",
    )

    parser.add_argument(
        "--simulate-power-budget",
        action="store_true",
        help="Simulate MAXN/50W/30W with latency budgets when not switching real hardware power mode",
    )
    parser.add_argument(
        "--apply-nvpmodel",
        action="store_true",
        help="Run `sudo nvpmodel -m <id>` before each power mode block",
    )
    parser.add_argument("--nvpmodel-maxn", type=int, default=None, help="nvpmodel id for MAXN")
    parser.add_argument("--nvpmodel-50w", type=int, default=None, help="nvpmodel id for 50W")
    parser.add_argument("--nvpmodel-30w", type=int, default=None, help="nvpmodel id for 30W")
    parser.add_argument(
        "--with-power",
        action="store_true",
        help="Run per-configuration power logging via monitor_power.py (requires jetson-stats/jtop).",
    )
    parser.add_argument(
        "--power-interval",
        type=float,
        default=0.5,
        help="Sampling interval for power logger (seconds).",
    )
    parser.add_argument(
        "--power-warmup-sec",
        type=float,
        default=1.0,
        help="Delay after power logger starts before processing begins.",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Enable tracking metrics (ByteTrack in this Python runner).",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "nvdcf"],
        help="Tracker to evaluate. NvDCF expects external MOT tracks exported from DeepStream.",
    )
    parser.add_argument(
        "--external-tracks-template",
        default="",
        help=(
            "Path template for external MOT tracks (used with --tracker nvdcf), "
            "e.g. /path/{run_id}_{power_mode}.txt"
        ),
    )
    parser.add_argument(
        "--tracker-frame-rate",
        type=int,
        default=30,
        help="Nominal frame rate used by ByteTrack.",
    )
    parser.add_argument(
        "--track-activation-threshold",
        type=float,
        default=0.25,
        help="ByteTrack activation threshold.",
    )
    parser.add_argument(
        "--track-matching-threshold",
        type=float,
        default=0.8,
        help="ByteTrack matching threshold.",
    )
    parser.add_argument(
        "--tracking-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for GT/pred matching in IDSW computation.",
    )
    parser.add_argument(
        "--vehicle-only-tracking",
        action="store_true",
        default=True,
        help="Track only COCO vehicle classes (default).",
    )
    parser.add_argument(
        "--all-classes-tracking",
        action="store_false",
        dest="vehicle_only_tracking",
        help="Disable vehicle-only tracking filter.",
    )
    parser.add_argument(
        "--gt-mot",
        default="",
        help="Optional MOT-format GT CSV/TXT file: frame,id,x,y,w,h,... for IDSW.",
    )
    parser.add_argument(
        "--gt-det",
        default="",
        help="Optional GT detections for mAP@0.5 (MOT-like CSV/TXT: frame,id,x,y,w,h,class?).",
    )
    parser.add_argument(
        "--gt-count",
        default="",
        help="Optional interval-count GT CSV with columns: condition,total_count (day/dusk/night).",
    )
    parser.add_argument(
        "--vehicle-only-detection-eval",
        action="store_true",
        default=True,
        help="Restrict detection mAP evaluation to vehicle classes when class labels are present.",
    )
    parser.add_argument(
        "--all-classes-detection-eval",
        action="store_false",
        dest="vehicle_only_detection_eval",
        help="Disable vehicle-only restriction for mAP evaluation.",
    )

    args = parser.parse_args()
    args.zero_dce_module = args.zero_dce_module.strip() or None
    args.derain_module = args.derain_module.strip() or None
    args.zero_dce_weights = args.zero_dce_weights.strip() or None
    args.derain_weights = args.derain_weights.strip() or None
    args.enhancer_device = args.enhancer_device.strip() or None
    args.video = args.video.strip()
    args.video_day = args.video_day.strip()
    args.video_dusk = args.video_dusk.strip()
    args.video_night = args.video_night.strip()
    args.gt_mot = args.gt_mot.strip() or None
    args.gt_det = args.gt_det.strip() or None
    args.gt_count = args.gt_count.strip() or None
    args.external_tracks_template = args.external_tracks_template.strip() or None
    args.p3_lowlight_conditions = {
        token.strip().lower()
        for token in args.p3_lowlight_conditions.split(",")
        if token.strip()
    }
    if not args.p3_lowlight_conditions:
        args.p3_lowlight_conditions = {"night"}
    if not (0.0 <= float(args.tau_night) <= 1.0):
        raise ValueError("--tau-night must be in [0, 1].")
    if args.enable_tracking and args.tracker == "nvdcf" and not args.external_tracks_template:
        raise ValueError("--tracker nvdcf requires --external-tracks-template.")
    return args


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    day_path = Path(args.video_day).resolve()
    dusk_path = Path(args.video_dusk).resolve()
    night_path = Path(args.video_night).resolve()

    print("=" * 64)
    print("IEEE Stressor Experiment Runner")
    print("=" * 64)
    print(f"Video mode: {args.video_mode}")
    if args.video_mode == "single":
        print(f"Video: {video_path}")
    else:
        print(f"Video day:   {day_path}")
        print(f"Video dusk:  {dusk_path}")
        print(f"Video night: {night_path}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print(f"Core runs: {len(build_core_runs(args.nmsfree_detector))} (R1-R5)")
    print(f"NMS-free detector slot: {args.nmsfree_detector}")
    print(f"Power modes: {POWER_MODES}")
    print("=" * 64)

    runner = IEEEExperiment(args)
    runner.run_all()


if __name__ == "__main__":
    main()
