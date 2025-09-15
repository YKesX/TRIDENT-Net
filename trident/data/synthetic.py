"""Synthetic data helpers for TRIDENT-Net."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import torch


def generate_synthetic_batch(
    B: int = 2,
    T: int = 16,
    H: int = 720,
    W: int = 1280,
    # Legacy aliases
    batch_size: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a synthetic batch matching TRIDENT-Net contracts.

    Returns keys:
      - rgb: Float[B,3,T,H,W]
      - ir:  Float[B,1,T,H,W]
      - kin: Float[B,3,9]
      - labels: {'hit': Float[B,1], 'kill': Float[B,1]}
      - class_id: Long[B]
      - class_conf: Float[B,1]
      - meta: list of dicts with times_ms
    """
    # Resolve legacy args
    if batch_size is not None:
        B = batch_size
    if height is not None:
        H = height
    if width is not None:
        W = width

    rng = np.random.default_rng(123)
    rgb = torch.rand(B, 3, T, H, W, dtype=torch.float32)
    ir = torch.rand(B, 1, T, H, W, dtype=torch.float32)
    kin = torch.randn(B, 3, 9, dtype=torch.float32) * 0.1
    hit = torch.from_numpy((rng.random(B) > 0.5).astype(np.float32)).view(B, 1)
    kill = (hit * torch.from_numpy((rng.random(B) > 0.5).astype(np.float32)).view(B, 1)).to(torch.float32)
    class_id = torch.from_numpy(rng.integers(0, 10, size=B)).to(torch.long)
    class_conf = torch.rand(B, 1, dtype=torch.float32)
    times = [{"shoot_ms": 1500, "hit_ms": 2000 if h.item() > 0.5 else None, "kill_ms": 3500 if (h.item() > 0.5 and k.item() > 0.5) else None} for h, k in zip(hit, kill)]

    # Modern keys
    out: Dict[str, Any] = {
        "rgb": rgb,
        "ir": ir,
        "kin": kin,
        "labels": {"hit": hit, "kill": kill},
        "class_id": class_id,
        "class_conf": class_conf,
        "meta": [{"times_ms": t} for t in times],
    }
    # Legacy/test keys
    out.update({
        "rgb_frames": rgb,
        "ir_frames": ir,
        "radar_data": torch.randn(B, 128, T),
        "kinematic_features": torch.randn(B, 384),
        "shoot": (hit > 0).squeeze(1).float(),
        "hit": hit.squeeze(1),
        "kill": kill.squeeze(1),
        "times_ms": {"pre": 1200, "fire": 700, "post": 1700},
        "batch_size": torch.tensor(B),
    })
    return out


def synthetic_jsonl(path: str, video_root: str, n: int = 8) -> None:
    """Write a small JSONL file for smoke tests.

    Each line contains minimal fields: scenario, video paths, and optional fps.
    Video files are not created here; consumers may replace paths.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i in range(n):
        stem = f"vid_{i:03d}"
        rec = {
            "scenario": ["miss", "hit", "kill"][i % 3],
            "video": {
                "path": f"{stem}.mp4",
                # Optional: omit IR to exercise zero-IR path
                # "ir_path": f"ir/{stem}_ir.mp4",
                "fps": 24,
            },
            "target": {
                "class_id": i % 5,
                "class_conf": float((i % 10) / 10.0),
            },
        }
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines), encoding="utf-8")


class SyntheticVideoJsonl:
    """Placeholder class for registry compatibility.

    Some configs reference data.synthetic.SyntheticVideoJsonl as a class; provide a
    minimal instantiable shell to satisfy dynamic creation, though the primary
    utilities are the functions above.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - not used in tests
        pass
