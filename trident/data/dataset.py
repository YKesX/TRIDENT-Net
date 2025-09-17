"""Dataset and loader utilities for TRIDENT-Net.

Implements a JSONL-backed video dataset producing synchronized RGB/IR clips with
timing windows and optional kinematics and class metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .video_ring import VideoRing
from .transforms import AlbuStereoClip
from .collate import pad_tracks_collate


logger = logging.getLogger(__name__)


NATIVE_W = 1280
NATIVE_H = 720


@dataclass
class DataFields:
    """Key mapping for JSONL records.

    Defaults match tasks.yml expectations but can be overridden per-config.
    """

    rgb_path_key: str = "video.path"
    ir_path_key: str = "video.ir_path"
    fps_key: str = "video.fps"
    kinematics_key: str = "radar.kinematics"
    class_id_key: str = "target.class_id"
    class_conf_key: str = "target.class_conf"
    prompt_key: str = "prompt"


def _dict_get(d: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _derive_labels(record: Dict[str, Any]) -> Dict[str, float]:
    # Prefer explicit labels if provided
    labels = _dict_get(record, "labels", {}) or {}
    if "hit" in labels and "kill" in labels:
        hit = float(labels["hit"])  # type: ignore[arg-type]
        kill = float(labels["kill"])  # type: ignore[arg-type]
        return {"hit": float(hit), "kill": float(kill)}

    scenario = record.get("scenario", "").lower()
    mapping = {
        "kill": (1.0, 1.0),
        "hit": (1.0, 0.0),
        "miss": (0.0, 0.0),
    }
    hit, kill = mapping.get(scenario, (0.0, 0.0))
    return {"hit": hit, "kill": kill}


def _default_times(record: Dict[str, Any]) -> Dict[str, int]:
    # If present use; otherwise use defaults
    shoot = record.get("shoot_ms")
    hit = record.get("hit_ms")
    kill = record.get("kill_ms")
    out = {"shoot_ms": int(shoot) if shoot is not None else 1500}
    if hit is not None:
        out["hit_ms"] = int(hit)
    else:
        if _derive_labels(record)["hit"] > 0.5:
            out["hit_ms"] = 2000
    if kill is not None:
        out["kill_ms"] = int(kill)
    else:
        if _derive_labels(record)["kill"] > 0.5:
            out["kill_ms"] = 3500
    return out


def _resolve_path(root: Path, p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    return path


class VideoJsonlDataset(Dataset):
    """JSONL video dataset producing synchronized RGB/IR clips.

    Output dict keys and shapes:
      - rgb: Float[3, T, 720, 1280]
      - ir: Float[1, T, 720, 1280]
      - kin: Float[3, 9]
      - labels: {hit: Float[1], kill: Float[1]}
      - class_id: Optional[Long[1]]
      - class_conf: Optional[Float[1]]
      - meta: Dict[str, Any]
    """

    def __init__(
        self,
        jsonl_path: str,
        video_root: str,
        preprocess: Dict[str, Any],
        fields_map: Optional[Dict[str, str]] | None = None,
        clip_sampler: Optional[Dict[str, Any]] | None = None,
        transforms_cfg: Optional[Dict[str, Any]] | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.video_root = Path(video_root)
        self.preprocess = preprocess
        # Tolerate extra/unknown keys in fields_map by filtering to known DataFields
        fm = fields_map or {}
        try:
            valid = {f.name for f in dataclasses.fields(DataFields)}
            fm_filtered = {k: v for k, v in fm.items() if k in valid}
            self.fields = DataFields(**fm_filtered)
        except Exception:
            logger.warning("Invalid fields_map provided; using defaults and ignoring unknown keys.")
            self.fields = DataFields()
        self.clip_sampler = clip_sampler or {}

        # Parse JSONL once
        self.samples: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        # Build transforms
        tcfg = transforms_cfg or {}
        norm = self.preprocess.get("normalize", {})
        imgsize = self.preprocess.get("image_size", {"h": NATIVE_H, "w": NATIVE_W})
        rgb_ops = (tcfg.get("rgb") or [])
        ir_ops = (tcfg.get("ir") or [])
        temporal = (tcfg.get("temporal") or {})
        self.transforms = AlbuStereoClip(rgb_ops=rgb_ops, ir_ops=ir_ops, temporal=temporal, image_size=imgsize, normalize=norm)

        # Window durations
        tw = self.preprocess.get("temporal_windows_ms", {})
        self.pre_ms = int(tw.get("pre_ms", 1200))
        self.fire_ms = int(tw.get("fire_ms", 700))
        self.post_ms = int(tw.get("post_ms", 1700))

        self.fps_assumed = int(self.preprocess.get("fps_assumed", 24))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        fields = self.fields

        rgb_rel = _dict_get(rec, fields.rgb_path_key)
        ir_rel = _dict_get(rec, fields.ir_path_key)
        fps = _dict_get(rec, fields.fps_key, self.fps_assumed)

        rgb_path = _resolve_path(self.video_root, rgb_rel)
        if rgb_path is None:
            raise ValueError(f"Missing RGB path in record {idx}")
        ir_path = _resolve_path(self.video_root, ir_rel) if ir_rel else None

        # Video rings
        rgb_ring = VideoRing(str(rgb_path), fps_hint=fps)
        rgb_ring.load_all()
        if ir_path and ir_path.exists():
            ir_ring = VideoRing(str(ir_path), fps_hint=fps)
            ir_ring.load_all()
        else:
            logger.warning(f"IR path missing for record {idx}, will use zeros IR")
            ir_ring = None

        times = _default_times(rec)

        # Slice windows
        rgb_slices = rgb_ring.freeze_and_slice(self.pre_ms, self.fire_ms, self.post_ms, t0_ms=times.get("shoot_ms", 0))
        if ir_ring is not None:
            ir_slices = ir_ring.freeze_and_slice(self.pre_ms, self.fire_ms, self.post_ms, t0_ms=times.get("shoot_ms", 0))
        else:
            # Make zero arrays matching rgb lengths
            total_T = sum(len(v) for v in rgb_slices.values())
            ir_zero = np.zeros((total_T, NATIVE_H, NATIVE_W, 1), dtype=np.uint8)
            ir_seq_np = ir_zero
            # Build rgb concat
            rgb_seq_np = np.concatenate([np.stack(rgb_slices[k], axis=0) for k in ("pre", "fire", "post")], axis=0)
            rgb_t, ir_t = self.transforms(rgb_seq_np, ir_seq_np)
            kin = self._load_kinematics(rec)
            labels = _derive_labels(rec)
            out: Dict[str, Any] = {
                "rgb": rgb_t,
                "ir": ir_t,
                "kin": kin,
                "labels": {"hit": torch.tensor([labels["hit"]], dtype=torch.float32), "kill": torch.tensor([labels["kill"]], dtype=torch.float32)},
                "meta": {
                    "rgb_path": str(rgb_path),
                    "ir_path": str(ir_path) if ir_path else None,
                    "fps": fps,
                    "times_ms": times,
                    "scenario": rec.get("scenario"),
                },
            }
            # Optional class
            class_id = _dict_get(rec, self.fields.class_id_key)
            if class_id is not None:
                out["class_id"] = torch.tensor([int(class_id)], dtype=torch.long)
            class_conf = _dict_get(rec, self.fields.class_conf_key)
            if class_conf is not None:
                out["class_conf"] = torch.tensor([float(class_conf)], dtype=torch.float32)
            return out

        # Both present: build sequences and apply transforms
        rgb_seq_np = np.concatenate([np.stack(rgb_slices[k], axis=0) for k in ("pre", "fire", "post")], axis=0)
        ir_seq_np = np.concatenate([np.stack(ir_slices[k], axis=0) for k in ("pre", "fire", "post")], axis=0)

        rgb_t, ir_t = self.transforms(rgb_seq_np, ir_seq_np)

        kin = self._load_kinematics(rec)
        labels = _derive_labels(rec)

        out2: Dict[str, Any] = {
            "rgb": rgb_t,
            "ir": ir_t,
            "kin": kin,
            "labels": {"hit": torch.tensor([labels["hit"]], dtype=torch.float32), "kill": torch.tensor([labels["kill"]], dtype=torch.float32)},
            "meta": {
                "rgb_path": str(rgb_path),
                "ir_path": str(ir_path) if ir_path else None,
                "fps": fps,
                "times_ms": times,
                "scenario": rec.get("scenario"),
            },
        }
        # Optional class
        class_id = _dict_get(rec, self.fields.class_id_key)
        if class_id is not None:
            out2["class_id"] = torch.tensor([int(class_id)], dtype=torch.long)
        class_conf = _dict_get(rec, self.fields.class_conf_key)
        if class_conf is not None:
            out2["class_conf"] = torch.tensor([float(class_conf)], dtype=torch.float32)

        return out2

    def _load_kinematics(self, rec: Dict[str, Any]) -> torch.Tensor:
        kin = _dict_get(rec, self.fields.kinematics_key)
        if kin is None:
            # Synthesize neutral small values
            return torch.zeros(3, 9, dtype=torch.float32)
        kin_arr = np.asarray(kin, dtype=np.float32)
        if kin_arr.shape != (3, 9):
            raise ValueError(f"Kinematics must be [3,9], got {kin_arr.shape}")
        return torch.from_numpy(kin_arr)


def _split_indices(n: int, train_ratio: float = 0.9) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    split = int(round(n * train_ratio))
    return idxs[:split], idxs[split:]


def create_data_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train/eval DataLoaders from configuration dict.

    Expects cfg like tasks.yml top-level keys: data, preprocess, etc.
    """
    data_cfg = cfg.get("data", {})
    sources = data_cfg.get("sources", {})
    jsonl_path = sources.get("jsonl_path")
    video_root = sources.get("video_root", "")

    dataset_cfg = data_cfg.get("dataset", {})
    fields_map = dataset_cfg.get("fields_map")
    clip_sampler = dataset_cfg.get("clip_sampler")
    transforms_cfg = dataset_cfg.get("transforms", {})

    preprocess = cfg.get("preprocess", {})

    dataset = VideoJsonlDataset(
        jsonl_path=jsonl_path,
        video_root=video_root,
        preprocess=preprocess,
        fields_map=fields_map,
        clip_sampler=clip_sampler,
        transforms_cfg=transforms_cfg,
    )

    # Split train/eval if desired; for simplicity, 90/10
    n = len(dataset)
    if n <= 1:
        train_ds = dataset
        val_ds = None
    else:
        train_idx, val_idx = _split_indices(n, train_ratio=0.9)
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)

    loader_cfg = data_cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 2))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))

    collate_fn = pad_tracks_collate

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader
