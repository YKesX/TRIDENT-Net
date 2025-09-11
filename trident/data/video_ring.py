"""
Real-time video ring buffer for TRIDENT-Net.

Implements VideoRing class for temporal window extraction from live video
streams at native 1280×720 resolution as specified in tasks.yml v0.4.1.

Author: Yağızhan Keskin
"""

import time
from typing import Optional, Tuple, Dict, Any
from collections import deque
import threading
import torch
import numpy as np


class VideoRing:
    """
    Ring buffer for real-time video capture and temporal window extraction.
    
    Maintains a circular buffer of video frames for efficient temporal
    window extraction during inference.
    """
    
    def __init__(
        self,
        seconds_capacity: float = 6.0,
        fps_hint: int = 24,
        device: str = "cpu"
    ) -> None:
        """
        Initialize video ring buffer.
        
        Args:
            seconds_capacity: Buffer capacity in seconds
            fps_hint: Expected frame rate for buffer sizing
            device: Device to store tensors on
        """
        self.seconds_capacity = seconds_capacity
        self.fps_hint = fps_hint
        self.device = device
        
        # Calculate buffer size
        self.max_frames = int(seconds_capacity * fps_hint)
        
        # Ring buffers for RGB and IR
        self.rgb_buffer = deque(maxlen=self.max_frames)
        self.ir_buffer = deque(maxlen=self.max_frames)
        self.timestamp_buffer = deque(maxlen=self.max_frames)
        
        # Synchronization
        self._lock = threading.RLock()
        self._frozen_rgb: Optional[torch.Tensor] = None
        self._frozen_ir: Optional[torch.Tensor] = None
        self._frozen_timestamps: Optional[np.ndarray] = None
        self._is_frozen = False
    
    def push_frame(
        self,
        rgb_frame: torch.Tensor,
        ir_frame: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add a new frame to the ring buffer.
        
        Args:
            rgb_frame: RGB frame tensor [3, H, W] at 1280×720
            ir_frame: Optional IR frame tensor [1, H, W] at 1280×720
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure correct shape (3, 720, 1280) for RGB
        if rgb_frame.shape != (3, 720, 1280):
            raise ValueError(f"RGB frame must be (3, 720, 1280), got {rgb_frame.shape}")
        
        if ir_frame is not None and ir_frame.shape != (1, 720, 1280):
            raise ValueError(f"IR frame must be (1, 720, 1280), got {ir_frame.shape}")
        
        with self._lock:
            # Don't add frames if frozen
            if self._is_frozen:
                return
            
            self.rgb_buffer.append(rgb_frame.to(self.device).clone())
            
            if ir_frame is not None:
                self.ir_buffer.append(ir_frame.to(self.device).clone())
            else:
                # Fill with zeros if no IR provided
                zero_ir = torch.zeros((1, 720, 1280), device=self.device, dtype=rgb_frame.dtype)
                self.ir_buffer.append(zero_ir)
            
            self.timestamp_buffer.append(timestamp)
    
    def capture(self, duration_seconds: float = 6.0) -> bool:
        """
        Start capturing frames for specified duration.
        
        Args:
            duration_seconds: How long to capture
            
        Returns:
            True if capture started successfully
        """
        # In a real implementation, this would interface with camera hardware
        # For now, we just ensure the buffer is ready
        with self._lock:
            return len(self.rgb_buffer) > 0
    
    def freeze_and_slice(
        self,
        pre_ms: int = 1200,
        fire_ms: int = 700, 
        post_ms: int = 1700,
        trigger_time: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Freeze buffer and extract temporal windows around trigger time.
        
        Args:
            pre_ms: Pre-fire window duration in milliseconds
            fire_ms: Fire window duration in milliseconds  
            post_ms: Post-fire window duration in milliseconds
            trigger_time: Trigger timestamp, defaults to current time
            
        Returns:
            Tuple of (rgb_seq, ir_seq) tensors [C, T, H, W]
        """
        if trigger_time is None:
            trigger_time = time.time()
        
        with self._lock:
            if len(self.rgb_buffer) == 0:
                # Return empty sequences if no frames
                return (
                    torch.zeros((3, 1, 720, 1280), device=self.device),
                    torch.zeros((1, 1, 720, 1280), device=self.device)
                )
            
            # Freeze current state
            self._frozen_rgb = torch.stack(list(self.rgb_buffer), dim=0)  # [T, 3, H, W]
            self._frozen_ir = torch.stack(list(self.ir_buffer), dim=0)    # [T, 1, H, W]  
            self._frozen_timestamps = np.array(list(self.timestamp_buffer))
            self._is_frozen = True
            
            # Extract temporal windows
            total_duration_ms = pre_ms + fire_ms + post_ms
            window_start = trigger_time - (pre_ms + fire_ms) / 1000.0
            window_end = trigger_time + post_ms / 1000.0
            
            # Find frames within window
            mask = (self._frozen_timestamps >= window_start) & (self._frozen_timestamps <= window_end)
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) == 0:
                # Use most recent frames if no frames in window
                rgb_seq = self._frozen_rgb[-1:].permute(1, 0, 2, 3)  # [C, T, H, W]
                ir_seq = self._frozen_ir[-1:].permute(1, 0, 2, 3)    # [C, T, H, W]
            else:
                # Extract frames in temporal order
                rgb_frames = self._frozen_rgb[valid_indices]  # [T, 3, H, W]
                ir_frames = self._frozen_ir[valid_indices]    # [T, 1, H, W]
                
                # Transpose to [C, T, H, W] format
                rgb_seq = rgb_frames.permute(1, 0, 2, 3)  # [3, T, H, W]
                ir_seq = ir_frames.permute(1, 0, 2, 3)    # [1, T, H, W]
                
                # Ensure minimum sequence length
                min_t = max(1, int(total_duration_ms * self.fps_hint / 1000 / 4))
                if rgb_seq.shape[1] < min_t:
                    # Pad by repeating last frame
                    pad_frames = min_t - rgb_seq.shape[1]
                    rgb_pad = rgb_seq[:, -1:].repeat(1, pad_frames, 1, 1)
                    ir_pad = ir_seq[:, -1:].repeat(1, pad_frames, 1, 1)
                    
                    rgb_seq = torch.cat([rgb_seq, rgb_pad], dim=1)
                    ir_seq = torch.cat([ir_seq, ir_pad], dim=1)
            
            return rgb_seq, ir_seq
    
    def unfreeze(self) -> None:
        """Resume normal buffer operation."""
        with self._lock:
            self._is_frozen = False
            self._frozen_rgb = None
            self._frozen_ir = None  
            self._frozen_timestamps = None
    
    def clear(self) -> None:
        """Clear all buffers."""
        with self._lock:
            self.rgb_buffer.clear()
            self.ir_buffer.clear()
            self.timestamp_buffer.clear()
            self.unfreeze()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            stats = {
                "frames_count": len(self.rgb_buffer),
                "capacity": self.max_frames,
                "usage_percent": len(self.rgb_buffer) / self.max_frames * 100,
                "is_frozen": self._is_frozen,
                "oldest_timestamp": self.timestamp_buffer[0] if self.timestamp_buffer else None,
                "newest_timestamp": self.timestamp_buffer[-1] if self.timestamp_buffer else None,
            }
            
            if len(self.timestamp_buffer) > 1:
                time_span = self.timestamp_buffer[-1] - self.timestamp_buffer[0]
                stats["time_span_seconds"] = time_span
                stats["effective_fps"] = len(self.timestamp_buffer) / max(time_span, 1e-6)
            
            return stats
    
    def __len__(self) -> int:
        """Return current number of frames in buffer."""
        with self._lock:
            return len(self.rgb_buffer)