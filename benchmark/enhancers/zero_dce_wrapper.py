#!/usr/bin/env python3
"""Zero-DCE enhancer wrapper for run_benchmark.py.

Interface expected by run_benchmark:
- configure(model_path: str | None, device: str | None, strict: bool) -> None
- enhance(frame_bgr: np.ndarray) -> np.ndarray
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ._torchscript_wrapper import TorchScriptImageEnhancer, choose_device, resolve_model_path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENGINE: Optional[TorchScriptImageEnhancer] = None
_CONFIGURED = False
_WARNED_FALLBACK = False


def configure(model_path: Optional[str] = None, device: Optional[str] = None, strict: bool = False) -> None:
    global _ENGINE, _CONFIGURED

    if _CONFIGURED:
        return

    resolved = resolve_model_path(
        explicit_path=model_path,
        env_var_name="ZERO_DCE_MODEL_PATH",
        default_relative_to_repo="models/zero_dce.torchscript.pt",
        repo_root=_REPO_ROOT,
    )
    dev = choose_device(device)

    if resolved and resolved.exists():
        _ENGINE = TorchScriptImageEnhancer(str(resolved), dev)
        print(f"[INFO] Zero-DCE enhancer loaded: {resolved} on {dev}")
    else:
        if strict:
            raise FileNotFoundError(
                "Zero-DCE weights not found. Provide --zero-dce-weights or ZERO_DCE_MODEL_PATH. "
                f"Checked: {resolved}"
            )
        _ENGINE = None

    _CONFIGURED = True


def _fallback_zero_dce_like(frame_bgr: np.ndarray) -> np.ndarray:
    """Heuristic curve-based low-light enhancement inspired by Zero-DCE behavior."""
    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        print("[WARN] Zero-DCE weights missing. Using heuristic curve fallback.")
        _WARNED_FALLBACK = True

    x = frame_bgr.astype(np.float32) / 255.0

    # Estimate scene darkness and pick a stronger enhancement curve for darker frames.
    luma = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean_luma = float(np.mean(luma))
    alpha = float(np.clip(0.55 - mean_luma, 0.12, 0.42))

    for _ in range(8):
        x = x + alpha * x * (1.0 - x)

    # Mild denoise + preserve edges.
    x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    out = cv2.bilateralFilter(x_u8, d=5, sigmaColor=35, sigmaSpace=35)
    return out


def enhance(frame_bgr: np.ndarray) -> np.ndarray:
    if not _CONFIGURED:
        configure()

    if _ENGINE is not None:
        return _ENGINE(frame_bgr)

    return _fallback_zero_dce_like(frame_bgr)
