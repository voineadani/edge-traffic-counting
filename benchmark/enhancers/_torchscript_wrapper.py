#!/usr/bin/env python3
"""Shared TorchScript image-enhancer utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


class TorchScriptImageEnhancer:
    """Runs image-to-image TorchScript models with [0,1] RGB tensor I/O."""

    def __init__(self, model_path: str, device: str) -> None:
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.inference_mode():
            out = self.model(tensor)

        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.detach().float().squeeze(0)

        # Support either [0,1] or [-1,1] outputs from restoration models.
        out_min = float(out.min().item())
        out_max = float(out.max().item())
        if out_min < 0.0:
            out = (out + 1.0) * 0.5
        if out_max > 1.5:
            out = out / 255.0

        out = out.clamp(0.0, 1.0)
        out = out.permute(1, 2, 0).cpu().numpy()
        out_u8 = (out * 255.0).astype(np.uint8)
        return cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)


def choose_device(explicit_device: Optional[str]) -> str:
    if explicit_device:
        return explicit_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_model_path(
    explicit_path: Optional[str],
    env_var_name: str,
    default_relative_to_repo: str,
    repo_root: Path,
) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        return p

    env_val = os.environ.get(env_var_name)
    if env_val:
        return Path(env_val).expanduser().resolve()

    fallback = (repo_root / default_relative_to_repo).resolve()
    return fallback
