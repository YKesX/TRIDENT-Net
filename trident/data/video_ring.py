"""Video ring buffer utilities.

Implements a CPU-side ring buffer for short video clips using OpenCV.

Contract
--------
- Accepts path to a video file.
- Decodes all frames into RAM on first use (clips are short ~8s).
- Validates native resolution 1280x720; raises ValueError otherwise.
- Provides temporal slicing around an anchor time in milliseconds and returns
  three contiguous segments: pre, fire, post.

Shapes
------
- Frames are numpy arrays with shape (H, W, C) where C is 3 (BGR) or 1 (IR).

Notes
-----
- OpenCV reads BGR by default; no conversion is performed here.
- Consumers can convert to RGB later (e.g., in transforms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


NATIVE_W = 1280
NATIVE_H = 720


@dataclass
class _Meta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


class VideoRing:
    """
    CPU ring buffer for video decoding with OpenCV.

    Parameters
    ----------
    path:
        Path to a video file readable by OpenCV.
    fps_hint:
        Optional hint for FPS; used when metadata is unreliable.
    capacity_seconds:
        Optional capacity in seconds; currently informational since we load all frames.

    Properties
    ----------
    width, height, fps, frames
        Video metadata and decoded frames.
    """

    def __init__(
        self,
        path: str,
        fps_hint: Optional[int] | None = None,
        capacity_seconds: Optional[float] | None = None,
    ) -> None:
        self._path = path
        self._fps_hint = fps_hint
        self._capacity_seconds = capacity_seconds
        self._meta: Optional[_Meta] = None
        self._frames: List[np.ndarray] = []

        self._capture: Optional[cv2.VideoCapture] = None
        self._open_capture()
        self._read_meta()

    # ---------------------------- internal utils ----------------------------
    def _open_capture(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self._path}")
        self._capture = cap

    def _read_meta(self) -> None:
        assert self._capture is not None
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = float(self._fps_hint or 24)

        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate native resolution
        if width != NATIVE_W or height != NATIVE_H:
            raise ValueError(
                f"Expected native 1280x720 video, got {width}x{height} for {self._path}"
            )

        self._meta = _Meta(
            path=self._path,
            fps=float(fps),
            frame_count=frame_count,
            width=width,
            height=height,
        )

    # ------------------------------ public API ------------------------------
    @property
    def width(self) -> int:
        """Native width (pixels)."""
        assert self._meta is not None
        return self._meta.width

    @property
    def height(self) -> int:
        """Native height (pixels)."""
        assert self._meta is not None
        return self._meta.height

    @property
    def fps(self) -> float:
        """Frames per second."""
        assert self._meta is not None
        return self._meta.fps

    @property
    def frames(self) -> List[np.ndarray]:
        """Decoded frames (BGR or single-channel)."""
        return self._frames

    def load_all(self) -> None:
        """Decode the entire clip into memory.

        Ensures frames are either 3-channel BGR or single-channel grayscale.
        """
        if self._frames:
            return
        assert self._capture is not None
        cap = self._capture
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            # Normalize to either (H,W,3) BGR or (H,W,1)
            if frame.ndim == 2:
                frame = frame[:, :, None]
            elif frame.ndim == 3 and frame.shape[2] == 3:
                pass
            elif frame.ndim == 3 and frame.shape[2] == 4:
                # Drop alpha if any
                frame = frame[:, :, :3]
            else:
                raise ValueError(f"Unexpected frame shape: {frame.shape}")
            frames.append(frame)
        self._frames = frames

    # Utility: ms to frame index
    def _ms_to_index(self, ms: int) -> int:
        assert self._meta is not None
        idx = int(round((ms / 1000.0) * self._meta.fps))
        return idx

    def _slice_indices(self, start_idx: int, end_idx: int) -> List[int]:
        assert self._meta is not None
        start = max(0, start_idx)
        end = min(self._meta.frame_count - 1, end_idx)
        if start > end:
            start, end = end, end
        # Inclusive of end; we want exact count with replication handled by caller
        return list(range(start, end + 1))

    def freeze_and_slice(
        self,
        pre_ms: int,
        fire_ms: int,
        post_ms: int,
        t0_ms: Optional[int] | None = 0,
    ) -> Dict[str, List[np.ndarray]]:
        """Slice the decoded frames into pre, fire, and post windows.

        Parameters
        ----------
        pre_ms, fire_ms, post_ms:
            Duration of each segment in milliseconds.
        t0_ms:
            Anchor time in milliseconds interpreted as the "shoot" moment.

        Returns
        -------
        dict
            Mapping of segment names to frame lists: {"pre": [...], "fire": [...], "post": [...]}.
            Each list length matches the requested duration in frames, with edge padding by replication.
        """
        if not self._frames:
            self.load_all()
        assert self._meta is not None

        fps = self._meta.fps
        ms_to_idx = lambda ms: int(round((ms / 1000.0) * fps))

        # Segment lengths in frames
        n_pre = ms_to_idx(pre_ms)
        n_fire = ms_to_idx(fire_ms)
        n_post = ms_to_idx(post_ms)

        t0_idx = ms_to_idx(t0_ms or 0)

        # Raw ranges
        pre_start = t0_idx - n_pre
        pre_end = t0_idx
        fire_start = t0_idx
        fire_end = t0_idx + n_fire
        post_start = fire_end
        post_end = fire_end + n_post

        # Get clamped indices
        pre_inds = self._slice_indices(pre_start, pre_end - 1) if n_pre > 0 else []
        fire_inds = self._slice_indices(fire_start, fire_end - 1) if n_fire > 0 else []
        post_inds = self._slice_indices(post_start, post_end - 1) if n_post > 0 else []

        # Pad by replication to ensure exact counts
        def take_with_pad(idxs: List[int], target: int) -> List[np.ndarray]:
            if target == 0:
                return []
            if not idxs:
                # No frames available; replicate first/last as zeros length? use first frame
                filler = self._frames[0] if self._frames else np.zeros((NATIVE_H, NATIVE_W, 3), dtype=np.uint8)
                return [filler.copy() for _ in range(target)]
            frames = [self._frames[i] for i in idxs]
            while len(frames) < target:
                frames.append(frames[-1])
            return frames

        out = {
            "pre": take_with_pad(pre_inds, n_pre),
            "fire": take_with_pad(fire_inds, n_fire),
            "post": take_with_pad(post_inds, n_post),
        }
        return out
