import argparse
import csv
import re
import time
from typing import Any

try:
    from jtop import jtop
except ImportError as exc:
    raise SystemExit(
        "jtop not found. Install jetson-stats (pip install jetson-stats) to use power logging."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson power logger")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=0.0)
    parser.add_argument(
        "--debug-keys",
        action="store_true",
        help="Print a one-time key snapshot for stats/power objects.",
    )
    return parser.parse_args()


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_NORM_RE = re.compile(r"[^a-z0-9]+")
_CPU_CORE_KEY_RE = re.compile(r"^cpu\d+$")


def _norm_key(key: Any) -> str:
    return _NORM_RE.sub("", str(key).lower())


def _to_float(value: Any, *, quantity: str) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    lower = text.lower().replace(",", "")
    match = _NUM_RE.search(lower)
    if not match:
        return None
    number = float(match.group(0))

    if quantity == "power_mw":
        if "kw" in lower:
            return number * 1_000_000.0
        if "mw" in lower:
            return number
        if lower.endswith("w") or " w" in lower:
            return number * 1000.0
        # Heuristic for unitless values: small numbers are usually Watts.
        if 0.0 < number <= 300.0:
            return number * 1000.0
        return number

    if quantity == "ram_mb":
        if "gb" in lower:
            return number * 1024.0
        if "kb" in lower:
            return number / 1024.0
        return number

    return number


def _pick_case_insensitive(data: dict[str, Any], keys: list[str]) -> Any:
    mapping = {_norm_key(k): v for k, v in data.items()}
    for key in keys:
        norm = _norm_key(key)
        if norm in mapping:
            return mapping[norm]
    return None


def _extract_power_entry_mw(entry: Any) -> float | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        direct = _pick_case_insensitive(entry, ["power", "tot", "total", "avg", "cur", "curr", "instant"])
        direct_val = _to_float(direct, quantity="power_mw")
        if direct_val is not None:
            return direct_val
        for value in entry.values():
            nested = _to_float(value, quantity="power_mw")
            if nested is not None:
                return nested
        return None
    return _to_float(entry, quantity="power_mw")


def _extract_power_total_mw(stats: dict[str, Any], power_obj: Any) -> float | None:
    # 1) Known total-power keys in stats.
    stats_direct = _pick_case_insensitive(
        stats,
        [
            "Power Tot",
            "Power TOT",
            "Power Total",
            "power_total_mw",
            "POM_5V_IN",
            "VDD_IN",
        ],
    )
    power = _extract_power_entry_mw(stats_direct)
    if power is not None:
        return power

    # 2) Keys that contain "power" and look like totals.
    for key, value in stats.items():
        norm = _norm_key(key)
        if "power" in norm and ("tot" in norm or "total" in norm or "in" in norm):
            power = _extract_power_entry_mw(value)
            if power is not None:
                return power

    # 3) jtop power object (varies by version).
    if isinstance(power_obj, dict):
        power_total = _pick_case_insensitive(
            power_obj,
            ["tot", "total", "power_tot", "power_total", "POM_5V_IN", "VDD_IN"],
        )
        power = _extract_power_entry_mw(power_total)
        if power is not None:
            return power

        for key, value in power_obj.items():
            norm = _norm_key(key)
            if "tot" in norm or "total" in norm or "pom5vin" in norm or "vddin" in norm:
                power = _extract_power_entry_mw(value)
                if power is not None:
                    return power

    return None


def _extract_gpu_load_pct(stats: dict[str, Any]) -> float | None:
    gpu = _pick_case_insensitive(stats, ["GPU", "gpu", "gpu_load", "gpu_load_pct"])
    if isinstance(gpu, dict):
        gpu = _pick_case_insensitive(gpu, ["load", "usage", "total", "percent", "value"])
    return _to_float(gpu, quantity="pct")


def _extract_cpu_load_pct(stats: dict[str, Any]) -> float | None:
    cpu = _pick_case_insensitive(stats, ["CPU", "cpu", "cpu_load", "cpu_load_pct"])
    if isinstance(cpu, dict):
        total = _pick_case_insensitive(cpu, ["Total", "total", "usage", "percent"])
        total_val = _to_float(total, quantity="pct")
        if total_val is not None:
            return total_val

        per_core_vals = []
        for value in cpu.values():
            parsed = _to_float(value, quantity="pct")
            if parsed is not None:
                per_core_vals.append(parsed)
        if per_core_vals:
            return sum(per_core_vals) / len(per_core_vals)
        return None

    cpu_val = _to_float(cpu, quantity="pct")
    if cpu_val is not None:
        return cpu_val

    # Some jtop versions expose only per-core keys at top level (CPU1..CPUn).
    per_core_vals = []
    for key, value in stats.items():
        if _CPU_CORE_KEY_RE.match(_norm_key(key)):
            parsed = _to_float(value, quantity="pct")
            if parsed is not None:
                per_core_vals.append(parsed)
    if per_core_vals:
        return sum(per_core_vals) / len(per_core_vals)
    return None


def _extract_ram_used_mb(stats: dict[str, Any]) -> float | None:
    ram = _pick_case_insensitive(stats, ["RAM", "ram", "ram_used_mb"])
    if isinstance(ram, dict):
        used = _pick_case_insensitive(ram, ["used", "use", "ram_used"])
        used_val = _to_float(used, quantity="ram_mb")
        if used_val is not None:
            return used_val
        return None

    return _to_float(ram, quantity="ram_mb")


def main() -> None:
    args = parse_args()

    with jtop() as jetson:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "wall_time_epoch",
                    "power_total_mw",
                    "gpu_load_pct",
                    "cpu_load_pct",
                    "ram_used_mb",
                ],
            )
            writer.writeheader()
            start = time.time()
            debug_dumped = False
            while jetson.ok():
                now = time.time()
                if args.max_duration > 0 and (now - start) >= args.max_duration:
                    break

                stats = jetson.stats
                power_obj = getattr(jetson, "power", None)
                power = _extract_power_total_mw(stats, power_obj)
                gpu = _extract_gpu_load_pct(stats)
                cpu = _extract_cpu_load_pct(stats)
                ram = _extract_ram_used_mb(stats)

                if args.debug_keys and not debug_dumped:
                    print(f"[DEBUG] stats keys: {sorted(stats.keys())}")
                    if isinstance(power_obj, dict):
                        print(f"[DEBUG] power keys: {sorted(power_obj.keys())}")
                    else:
                        print(f"[DEBUG] power object type: {type(power_obj).__name__}")
                    debug_dumped = True

                writer.writerow(
                    {
                        "wall_time_epoch": f"{now:.6f}",
                        "power_total_mw": f"{power:.3f}" if power is not None else "",
                        "gpu_load_pct": f"{gpu:.3f}" if gpu is not None else "",
                        "cpu_load_pct": f"{cpu:.3f}" if cpu is not None else "",
                        "ram_used_mb": f"{ram:.3f}" if ram is not None else "",
                    }
                )
                f.flush()
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
