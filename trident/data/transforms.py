"""
Synchronized RGB/IR transformations for TRIDENT-Net.

Implements AlbuStereoClip for synchronized augmentations across RGB and IR channels
with temporal jittering and dropout as specified in tasks.yml v0.4.1.

Author: Yağızhan Keskin
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import cv2


class AlbuStereoClip:
    """
    Synchronized RGB/IR transformations for temporal video clips.
    
    Applies augmentations consistently across RGB and IR channels while
    supporting temporal jittering and frame dropout.
    """
    
    def __init__(
        self,
        rgb: Optional[List[Dict[str, Any]]] = None,
        ir: Optional[List[Dict[str, Any]]] = None,
        temporal: Optional[Dict[str, Any]] = None,
        p: float = 0.5
    ) -> None:
        """
        Initialize synchronized transformations.
        
        Args:
            rgb: List of RGB augmentation configs
            ir: List of IR augmentation configs  
            temporal: Temporal jittering/dropout config
            p: Probability of applying augmentations
        """
        self.rgb_transforms = rgb or []
        self.ir_transforms = ir or []
        self.temporal_config = temporal or {}
        self.p = p
        
        # Parse temporal config
        self.jitter_frames = self.temporal_config.get('jitter_frames', 1)
        self.dropout_frames_p = self.temporal_config.get('dropout_frames_p', 0.05)
    
    def __call__(
        self, 
        rgb_seq: torch.Tensor, 
        ir_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized transformations.
        
        Args:
            rgb_seq: RGB sequence [C, T, H, W] 
            ir_seq: IR sequence [C, T, H, W]
            
        Returns:
            Tuple of transformed (rgb_seq, ir_seq)
        """
        if random.random() > self.p:
            return rgb_seq, ir_seq
        
        # Apply temporal jittering first
        rgb_seq, ir_seq = self._apply_temporal_jitter(rgb_seq, ir_seq)
        
        # Apply frame dropout
        rgb_seq, ir_seq = self._apply_frame_dropout(rgb_seq, ir_seq)
        
        # Apply spatial augmentations frame by frame
        rgb_seq = self._apply_spatial_augs(rgb_seq, self.rgb_transforms)
        ir_seq = self._apply_spatial_augs(ir_seq, self.ir_transforms)
        
        return rgb_seq, ir_seq
    
    def _apply_temporal_jitter(
        self, 
        rgb_seq: torch.Tensor, 
        ir_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal jittering by shifting frame indices."""
        if self.jitter_frames <= 0:
            return rgb_seq, ir_seq
        
        T = rgb_seq.shape[1]
        if T <= 2 * self.jitter_frames:
            return rgb_seq, ir_seq
        
        # Random shift within jitter range
        jitter = random.randint(-self.jitter_frames, self.jitter_frames)
        
        if jitter > 0:
            # Shift forward - pad with first frames
            rgb_seq = torch.cat([
                rgb_seq[:, :jitter].repeat(1, 1, 1, 1),
                rgb_seq[:, :-jitter]
            ], dim=1)
            ir_seq = torch.cat([
                ir_seq[:, :jitter].repeat(1, 1, 1, 1), 
                ir_seq[:, :-jitter]
            ], dim=1)
        elif jitter < 0:
            # Shift backward - pad with last frames
            rgb_seq = torch.cat([
                rgb_seq[:, -jitter:],
                rgb_seq[:, :jitter].repeat(1, 1, 1, 1)
            ], dim=1)
            ir_seq = torch.cat([
                ir_seq[:, -jitter:],
                ir_seq[:, :jitter].repeat(1, 1, 1, 1)
            ], dim=1)
        
        return rgb_seq, ir_seq
    
    def _apply_frame_dropout(
        self,
        rgb_seq: torch.Tensor,
        ir_seq: torch.Tensor  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random frame dropout."""
        if self.dropout_frames_p <= 0:
            return rgb_seq, ir_seq
        
        T = rgb_seq.shape[1]
        
        # Randomly select frames to drop
        drop_mask = torch.rand(T) < self.dropout_frames_p
        
        # Ensure we don't drop too many frames
        if drop_mask.sum() >= T - 1:
            drop_mask = torch.zeros(T, dtype=torch.bool)
            if T > 1:
                drop_mask[random.randint(0, T-1)] = True
        
        # Replace dropped frames with neighboring frames
        if drop_mask.any():
            keep_indices = torch.where(~drop_mask)[0]
            
            for i in torch.where(drop_mask)[0]:
                # Find nearest kept frame
                distances = torch.abs(keep_indices - i)
                nearest_idx = keep_indices[distances.argmin()]
                
                rgb_seq[:, i] = rgb_seq[:, nearest_idx].clone()
                ir_seq[:, i] = ir_seq[:, nearest_idx].clone()
        
        return rgb_seq, ir_seq
    
    def _apply_spatial_augs(
        self,
        seq: torch.Tensor, 
        transforms: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Apply spatial augmentations to sequence."""
        C, T, H, W = seq.shape
        
        # Convert to numpy for albumentations-style processing
        seq_np = seq.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
        
        # Apply each transform
        for transform_config in transforms:
            transform_name = transform_config.get('name')
            transform_p = transform_config.get('p', 1.0)
            
            if random.random() > transform_p:
                continue
            
            seq_np = self._apply_single_transform(seq_np, transform_config)
        
        # Convert back to tensor
        seq_tensor = torch.from_numpy(seq_np).permute(3, 0, 1, 2)  # [C, T, H, W]
        return seq_tensor.to(seq.device)
    
    def _apply_single_transform(
        self,
        seq_np: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply a single transformation type."""
        transform_name = config.get('name')
        T, H, W, C = seq_np.shape
        
        if transform_name == 'RandomBrightnessContrast':
            brightness_limit = config.get('brightness_limit', 0.2)
            contrast_limit = config.get('contrast_limit', 0.2)
            
            brightness = random.uniform(-brightness_limit, brightness_limit)
            contrast = random.uniform(1-contrast_limit, 1+contrast_limit)
            
            seq_np = np.clip(seq_np * contrast + brightness, 0, 1)
            
        elif transform_name == 'HueSaturationValue':
            hue_shift = config.get('hue_shift_limit', 8) / 180.0  # Convert to [0,1]
            sat_shift = config.get('sat_shift_limit', 12) / 100.0
            val_shift = config.get('val_shift_limit', 12) / 100.0
            
            if C == 3:  # Only for RGB
                for t in range(T):
                    frame = seq_np[t]
                    # Simple HSV adjustment approximation
                    frame = np.clip(frame + random.uniform(-val_shift, val_shift), 0, 1)
                    seq_np[t] = frame
        
        elif transform_name == 'MotionBlur':
            blur_limit = config.get('blur_limit', 5)
            kernel_size = random.randint(3, blur_limit)
            
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            for t in range(T):
                frame = (seq_np[t] * 255).astype(np.uint8)
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                seq_np[t] = frame.astype(np.float32) / 255.0
        
        elif transform_name == 'JPEGCompression':
            quality_lower = config.get('quality_lower', 70)
            quality_upper = config.get('quality_upper', 95)
            quality = random.randint(quality_lower, quality_upper)
            
            for t in range(T):
                frame = (seq_np[t] * 255).astype(np.uint8)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                _, encoded = cv2.imencode('.jpg', frame, encode_params)
                frame = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    seq_np[t] = frame.astype(np.float32) / 255.0
        
        elif transform_name == 'HorizontalFlip':
            if random.random() < 0.5:
                seq_np = np.flip(seq_np, axis=2)  # Flip width dimension
        
        elif transform_name == 'CLAHE':
            clip_limit = config.get('clip_limit', 2.0)
            tile_grid_size = config.get('tile_grid_size', [8, 8])
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
            
            for t in range(T):
                frame = (seq_np[t] * 255).astype(np.uint8)
                if C == 1:
                    frame = clahe.apply(frame[:, :, 0])
                    seq_np[t, :, :, 0] = frame.astype(np.float32) / 255.0
                else:
                    for c in range(C):
                        frame_c = clahe.apply(frame[:, :, c])
                        seq_np[t, :, :, c] = frame_c.astype(np.float32) / 255.0
        
        elif transform_name == 'GaussNoise':
            var_limit = config.get('var_limit', [10.0, 40.0])
            noise_var = random.uniform(var_limit[0], var_limit[1])
            noise = np.random.normal(0, noise_var/255.0, seq_np.shape)
            seq_np = np.clip(seq_np + noise, 0, 1)
        
        return seq_np