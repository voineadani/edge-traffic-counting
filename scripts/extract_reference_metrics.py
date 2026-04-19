#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    maxn_dir = ROOT / "benchmark" / "reference_results_maxn"
    full_dir = ROOT / "benchmark" / "results_full_power"
    nvdcf_summary = ROOT / "benchmark" / "reference_results_nvdcf_30w" / "summary.json"

    print("Reference benchmark metrics")
    print()

    print("R1-R5 MAXN latency / count summaries:")
    for run_id in ["R1", "R2", "R3", "R4", "R5"]:
        for cond in ["day", "dusk", "night"]:
            p = maxn_dir / f"{run_id}_MAXN_{cond}.json"
            if not p.exists():
                continue
            d = load_json(p)
            ce = d.get("count_eval", {})
            print(
                f"{run_id} {cond}: "
                f"lat={d.get('mean_latency_ms', 0):.2f} ms, "
                f"pred={ce.get('pred_total_count')}, "
                f"gt={ce.get('gt_total_count')}, "
                f"ape={ce.get('relative_error_pct'):.2f}%"
            )

    print()
    print("R1-R5 50W latency / count summaries:")
    for run_id in ["R1", "R2", "R3", "R4", "R5"]:
        for cond in ["day", "dusk", "night"]:
            p = full_dir / f"{run_id}_50W_{cond}.json"
            if not p.exists():
                continue
            d = load_json(p)
            ce = d.get("count_eval", {})
            print(
                f"{run_id} {cond}: "
                f"lat={d.get('mean_latency_ms', 0):.2f} ms, "
                f"pred={ce.get('pred_total_count')}, "
                f"gt={ce.get('gt_total_count')}, "
                f"ape={ce.get('relative_error_pct'):.2f}%"
            )

    print()
    if nvdcf_summary.exists():
        d = load_json(nvdcf_summary)
        print("NvDCF summary file present:")
        print(nvdcf_summary)
        print(f"runs={d.get('total_runs')} aggregated_runs={d.get('total_aggregated_runs')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
