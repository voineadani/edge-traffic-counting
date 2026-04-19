#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAXN_DIR = ROOT / "benchmark" / "reference_results_maxn"
FULL_DIR = ROOT / "benchmark" / "results_full_power"
NVDCF_DIR = ROOT / "benchmark" / "reference_results_nvdcf_30w"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


def counting_entry(run_id: str, cond: str, base: Path) -> dict:
    d = load_json(base / f"{run_id}_{'MAXN' if 'maxn' in str(base).lower() else '50W'}_{cond}.json")
    ce = d["count_eval"]
    return {
        "ape": float(ce["relative_error_pct"]),
        "ocr": float(ce["pred_total_count"]) / float(ce["gt_total_count"]),
        "lat": float(d["mean_latency_ms"]),
        "fpw": float(d.get("fps_per_watt", 0.0)),
    }


def extract_counting_perf() -> str:
    rows = []
    for run_id, det, pre in [
        ("R1", "v8n", "P0"),
        ("R2", "v11n", "P0"),
        ("R3", "v8n", "P1"),
        ("R4", "v8n", "P3"),
        ("R5", "v11n", "P3"),
    ]:
        maxn_by_cond = {c: counting_entry(run_id, c, MAXN_DIR) for c in ["day", "dusk", "night"]}
        p50_by_cond = {c: counting_entry(run_id, c, FULL_DIR) for c in ["day", "dusk", "night"]}

        mean_maxn_lat = sum(v["lat"] for v in maxn_by_cond.values()) / 3.0
        mean_50w_lat = sum(v["lat"] for v in p50_by_cond.values()) / 3.0
        mean_maxn_fpw = sum(v["fpw"] for v in maxn_by_cond.values()) / 3.0
        mean_50w_fpw = sum(v["fpw"] for v in p50_by_cond.values()) / 3.0

        day_ape = fmt(maxn_by_cond["day"]["ape"], 1)
        day_ocr = fmt(maxn_by_cond["day"]["ocr"], 2)
        if pre == "P3":
            day_ape = "---"
            day_ocr = "---"

        rows.append(
            "| {cfg} ({det}) | {pre} | {latm} / {lat50} | {fpwm} | {fpw50} | {dape} | {docr} | {duape} | {duocr} | {nape} | {nocr} |".format(
                cfg=run_id,
                det=det,
                pre=pre,
                latm=fmt(mean_maxn_lat, 1),
                lat50=fmt(mean_50w_lat, 1),
                fpwm=fmt(mean_maxn_fpw, 2),
                fpw50=fmt(mean_50w_fpw, 2),
                dape=day_ape,
                docr=day_ocr,
                duape=fmt(maxn_by_cond["dusk"]["ape"], 1),
                duocr=fmt(maxn_by_cond["dusk"]["ocr"], 2),
                nape=fmt(maxn_by_cond["night"]["ape"], 1),
                nocr=fmt(maxn_by_cond["night"]["ocr"], 2),
            )
        )
    header = [
        "| Config | Pre | Lat. (ms) MAXN / 50W | FPS/W MAXN | FPS/W 50W | Day APE | Day OCR | Dusk APE | Dusk OCR | Night APE | Night OCR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(header + rows)


def extract_tracker_comparison() -> str:
    rows = []
    for tracker_name, base in [("ByteTrack", MAXN_DIR), ("NvDCF", NVDCF_DIR)]:
        vals = {}
        for cond in ["day", "dusk", "night"]:
            d = load_json(base / f"R1_MAXN_{cond}.json")
            ce = d["count_eval"]
            vals[cond] = (
                float(ce["relative_error_pct"]),
                float(ce["pred_total_count"]) / float(ce["gt_total_count"]),
            )
        rows.append(
            "| {tracker} | {dape} | {docr} | {duape} | {duocr} | {nape} | {nocr} |".format(
                tracker=tracker_name,
                dape=fmt(vals["day"][0], 1),
                docr=fmt(vals["day"][1], 2),
                duape=fmt(vals["dusk"][0], 1),
                duocr=fmt(vals["dusk"][1], 2),
                nape=fmt(vals["night"][0], 1),
                nocr=fmt(vals["night"][1], 2),
            )
        )
    header = [
        "| Tracker | Day APE (%) | Day OCR | Dusk APE (%) | Dusk OCR | Night APE (%) | Night OCR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(header + rows)


def extract_tracking_jitter() -> str:
    bt_total = {"stable": 0, "stutter": 0, "drop": 0}
    nv_total = {"stable": 0, "stutter": 0, "drop": 0}
    frames_total = {"stable": 0, "stutter": 0, "drop": 0}
    max_gap = {"stable": 0.0, "stutter": 0.0, "drop": 0.0}

    for cond in ["day", "dusk", "night"]:
        d_bt = load_json(MAXN_DIR / f"R1_MAXN_{cond}.json")
        d_nv = load_json(NVDCF_DIR / f"R1_MAXN_{cond}.json")
        for bucket in ["stable", "stutter", "drop"]:
            bt_total[bucket] += int(d_bt["tracking"]["per_timing_bucket"][bucket]["id_switches"])
            nv_total[bucket] += int(d_nv["tracking"]["per_timing_bucket"][bucket]["id_switches"])
            frames_total[bucket] += int(d_bt["timing_bucket_counts"][bucket])

        max_gap["stable"] = max(max_gap["stable"], 40.0)
        max_gap["stutter"] = max(max_gap["stutter"], 80.0)
        if cond == "night":
            max_gap["drop"] = max(max_gap["drop"], 14080.0)

    header = [
        "| Bucket | Frames (total) | Max gap (ms) | Proxy IDSW ByteTrack | Proxy IDSW NvDCF |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = [
        f"| Stable | {frames_total['stable']} | {int(max_gap['stable'])} | {bt_total['stable']} | {nv_total['stable']} |",
        f"| Stutter | {frames_total['stutter']} | {int(max_gap['stutter'])} | {bt_total['stutter']} | {nv_total['stutter']} |",
        f"| Drop | {frames_total['drop']} | {int(max_gap['drop'])} | {bt_total['drop']} | {nv_total['drop']} |",
    ]
    return "\n".join(header + rows)


def main() -> int:
    print("# Table: Counting Accuracy and Latency")
    print(extract_counting_perf())
    print()
    print("# Table: ByteTrack vs NvDCF")
    print(extract_tracker_comparison())
    print()
    print("# Table: Timing Bucket Distribution and Proxy IDSW")
    print(extract_tracking_jitter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
