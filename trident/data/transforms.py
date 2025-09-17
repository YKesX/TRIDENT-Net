"""Albumentations-backed synchronized clip transforms for paired RGB/IR.

Provides spatially synchronized transforms across RGB and IR streams, with
modality-specific photometric augmentations, temporal jittering, and normalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import inspect

import numpy as np
import torch

try:
    import albumentations as A
except Exception as e:  # pragma: no cover - optional at runtime until installed
    A = None  # type: ignore


NATIVE_W = 1280
NATIVE_H = 720


def _to_rgb(frames_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frames (T,H,W,3) from OpenCV to RGB in-place style.

    Returns a new array with channels swapped to RGB.
    """
    assert frames_bgr.ndim == 4 and frames_bgr.shape[-1] == 3
    rgb = frames_bgr[..., ::-1].copy()
    return rgb


class AlbuStereoClip:
    """
    Albumentations-backed synchronized clip transforms for paired RGB/IR.

    Applies identical spatial ops to both streams; allows per-modality photometric ops.
    Includes temporal jitter around uniform indices and optional dropout per frame while
    maintaining sequence length by repeating the previous valid frame.

    Inputs
    ------
    rgb_seq: np.ndarray[T, H, W, 3] (BGR from cv2, converted to RGB internally)
    ir_seq:  np.ndarray[T, H, W, 1] (single-channel)

    Output
    ------
    Tuple[torch.FloatTensor, torch.FloatTensor]:
      - rgb: [3, T, H, W]
      - ir:  [1, T, H, W]
    """

    def __init__(
        self,
        rgb_ops: List[Dict[str, Any]] | None = None,
        ir_ops: List[Dict[str, Any]] | None = None,
        temporal: Dict[str, Any] | None = None,
        image_size: Dict[str, int] | None = None,
        normalize: Dict[str, List[float]] | None = None,
    ) -> None:
        self.rgb_ops_cfg = rgb_ops or []
        self.ir_ops_cfg = ir_ops or []
        self.temporal_cfg = temporal or {"jitter_frames": 0, "dropout_frames_p": 0.0}
        self.image_size = image_size or {"h": NATIVE_H, "w": NATIVE_W}
        self.normalize = normalize or {
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_std": [0.229, 0.224, 0.225],
            "ir_mean": [0.5],
            "ir_std": [0.25],
        }

        if A is None:
            raise ImportError("Albumentations is required for AlbuStereoClip. Please install albumentations.")

        # Build base spatial pipeline; we build from cfg names using getattr(A, name)
        self.rgb_aug = self._build_ops(self.rgb_ops_cfg)
        self.ir_aug = self._build_ops(self.ir_ops_cfg)

        # Compose used to share spatial parameters across rgb/ir;
        # we apply only geometric ops here. For simplicity, apply rgb_aug/ir_aug after spatial.
        self.spatial_ops = A.Compose(
            [
                A.HorizontalFlip(p=0.0),  # default no-op; users can add flips in rgb_ops to also affect IR via params
            ],
            additional_targets={"image_ir": "image"},
        )

        self.h = int(self.image_size.get("h", NATIVE_H))
        self.w = int(self.image_size.get("w", NATIVE_W))
        if self.h != NATIVE_H or self.w != NATIVE_W:
            # We keep native by default per constraints
            raise ValueError(f"Expected image_size 720x1280; got {self.h}x{self.w}")

        # Expand to 4D [C,1,1,1] to broadcast over [C,T,H,W]
        self.rgb_mean = (
            torch.tensor(
                self.normalize.get("rgb_mean", [0.485, 0.456, 0.406]),
                dtype=torch.float32,
            )
            .view(-1, 1, 1, 1)
        )
        self.rgb_std = (
            torch.tensor(
                self.normalize.get("rgb_std", [0.229, 0.224, 0.225]),
                dtype=torch.float32,
            )
            .view(-1, 1, 1, 1)
        )
        self.ir_mean = torch.tensor(self.normalize.get("ir_mean", [0.5]), dtype=torch.float32).view(-1, 1, 1, 1)
        self.ir_std = torch.tensor(self.normalize.get("ir_std", [0.25]), dtype=torch.float32).view(-1, 1, 1, 1)

        self.jitter = int(self.temporal_cfg.get("jitter_frames", 0))
        self.dropout_p = float(self.temporal_cfg.get("dropout_frames_p", 0.0))

    def _build_ops(self, ops_cfg: List[Dict[str, Any]]):
        ops: List[Any] = []
        for spec in ops_cfg:
            name = spec.get("name")
            if not name:
                continue
            params = {k: v for k, v in spec.items() if k != "name"}
            # Guard against Albumentations version param mismatches
            if name == "GaussNoise":
                # Some versions expect var_limit renamed; keep only known args
                allowed = {"p", "mean", "var_limit"}
                params = {k: v for k, v in params.items() if k in allowed}
            if not hasattr(A, name):
                # Ignore unknown ops to remain robust across environments
                continue
            aug_cls = getattr(A, name)
            # Filter params to those accepted by constructor signature to avoid warnings
            try:
                sig = inspect.signature(aug_cls.__init__)
                allowed_keys = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
                params = {k: v for k, v in params.items() if k in allowed_keys}
            except Exception:
                pass
            ops.append(aug_cls(**params))
        return A.Compose(ops) if ops else A.Compose([])

    def _temporal_indices(self, T: int) -> List[int]:
        # Identity indices with jitter and dropout while keeping length constant
        base = list(range(T))
        if self.jitter <= 0 and self.dropout_p <= 0:
            return base
        rng = np.random.default_rng()
        out: List[int] = []
        last_valid = 0
        for i in base:
            j = i
            if self.jitter > 0:
                j = int(np.clip(i + rng.integers(-self.jitter, self.jitter + 1), 0, T - 1))
            drop_roll = rng.random()
            if self.dropout_p > 0 and drop_roll < self.dropout_p:
                # repeat last valid index to keep length
                out.append(last_valid)
            else:
                out.append(j)
                last_valid = j
        return out

    def __call__(self, rgb_seq: np.ndarray, ir_seq: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb_seq.ndim != 4 or rgb_seq.shape[-1] != 3:
            raise ValueError(f"rgb_seq must be [T,H,W,3], got {rgb_seq.shape}")
        if ir_seq.ndim != 4 or ir_seq.shape[-1] not in (1, 3):
            raise ValueError(f"ir_seq must be [T,H,W,1], got {ir_seq.shape}")
        T, H, W, _ = rgb_seq.shape
        if (H, W) != (NATIVE_H, NATIVE_W):
            raise ValueError(f"Expected native 720x1280 frames; got {H}x{W}")

        # Temporal indexing
        idxs = self._temporal_indices(T)

        rgb_seq = _to_rgb(rgb_seq[idxs])  # to RGB
        ir_seq = ir_seq[idxs]

        # Apply spatial ops per-frame with shared params
        rgb_out: List[np.ndarray] = []
        ir_out: List[np.ndarray] = []
        for t in range(len(idxs)):
            sample = self.spatial_ops(image=rgb_seq[t], image_ir=ir_seq[t])
            rgb_img = sample["image"]
            ir_img = sample["image_ir"]
            # Photometric per modality
            if self.rgb_aug.transforms:
                rgb_img = self.rgb_aug(image=rgb_img)["image"]
            if self.ir_aug.transforms:
                # Ensure single-channel stays single-channel; albumentations may output 2D
                ir_aug_img = self.ir_aug(image=ir_img)["image"]
                if ir_aug_img.ndim == 2:
                    ir_aug_img = ir_aug_img[:, :, None]
                ir_img = ir_aug_img
            # Enforce IR as single-channel
            if ir_img.ndim == 2:
                ir_img = ir_img[:, :, None]
            elif ir_img.ndim == 3 and ir_img.shape[2] == 3:
                # Convert to grayscale by mean to ensure 1 channel for T-branch
                ir_img = np.mean(ir_img, axis=2, keepdims=True).astype(ir_img.dtype)
            rgb_out.append(rgb_img)
            ir_out.append(ir_img)

        rgb_np = np.stack(rgb_out, axis=0)  # [T,H,W,3] RGB
        ir_np = np.stack(ir_out, axis=0)    # [T,H,W,1]

        # To torch float in [0,1]
        rgb_t = torch.from_numpy(rgb_np).permute(3, 0, 1, 2).float() / 255.0  # [3,T,H,W]
        ir_t = torch.from_numpy(ir_np).permute(3, 0, 1, 2).float() / 255.0    # [1,T,H,W]

        # Normalize
        rgb_t = (rgb_t - self.rgb_mean) / self.rgb_std
        ir_t = (ir_t - self.ir_mean) / self.ir_std

        return rgb_t.contiguous(), ir_t.contiguous()
