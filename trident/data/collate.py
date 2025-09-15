"""Collate utilities for TRIDENT-Net data loaders."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _pad_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
    if x.dim() < 3:
        return x
    # x is [C, T, H, W] or [1, T, H, W]
    C, T, *rest = x.shape
    if T == target_T:
        return x
    pad_T = target_T - T
    pad_shape = (C, pad_T, *rest)
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def pad_tracks_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-T clip tensors in a batch and stack.

    Handles keys: 'rgb' [3,T,H,W], 'ir' [1,T,H,W], optional 'kin' [3,9],
    'labels' dict with 'hit'/'kill' Float[1], optional 'class_id' Long[1],
    'class_conf' Float[1], and 'meta'.
    """
    # Determine max T
    T_list: List[int] = []
    for sample in batch:
        if "rgb" in sample:
            T_list.append(sample["rgb"].shape[1])
        elif "ir" in sample:
            T_list.append(sample["ir"].shape[1])
    max_T = max(T_list) if T_list else 0

    # Pad and collect
    rgb_stack: List[torch.Tensor] = []
    ir_stack: List[torch.Tensor] = []
    kin_stack: List[torch.Tensor] = []
    y_hit: List[torch.Tensor] = []
    y_kill: List[torch.Tensor] = []
    class_ids: List[torch.Tensor] = []
    class_confs: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []

    for sample in batch:
        if "rgb" in sample:
            rgb_stack.append(_pad_time(sample["rgb"], max_T))
        if "ir" in sample:
            ir_stack.append(_pad_time(sample["ir"], max_T))
        if "kin" in sample:
            kin_stack.append(sample["kin"].to(torch.float32))
        if "labels" in sample:
            lab = sample["labels"]
            y_hit.append(lab["hit"].to(torch.float32).view(1))
            y_kill.append(lab["kill"].to(torch.float32).view(1))
        if "class_id" in sample:
            class_ids.append(sample["class_id"].to(torch.long).view(1))
        if "class_conf" in sample:
            class_confs.append(sample["class_conf"].to(torch.float32).view(1))
        if "meta" in sample:
            metas.append(sample["meta"])  # keep as dicts

    out: Dict[str, Any] = {}
    if rgb_stack:
        out["rgb"] = torch.stack(rgb_stack, dim=0)
    if ir_stack:
        out["ir"] = torch.stack(ir_stack, dim=0)
    if kin_stack:
        out["kin"] = torch.stack(kin_stack, dim=0)
    if y_hit:
        out.setdefault("labels", {})
        out["labels"]["hit"] = torch.stack(y_hit, dim=0)
    if y_kill:
        out.setdefault("labels", {})
        out["labels"]["kill"] = torch.stack(y_kill, dim=0)
    if class_ids:
        out["class_id"] = torch.stack(class_ids, dim=0).view(-1)
    if class_confs:
        out["class_conf"] = torch.stack(class_confs, dim=0)
    if metas:
        out["meta"] = metas

    return out
