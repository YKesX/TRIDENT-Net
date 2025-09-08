"""
TRIDENT-I1: Frag3D - 3D U-Net segmentation over 3 RGB frames

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken


class Conv3DBlock(nn.Module):
    """3D Convolution block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, norm: str = "group"):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        
        if norm == "group":
            self.norm = nn.GroupNorm(8, out_channels)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = nn.Identity()
            
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class Down3D(nn.Module):
    """3D downsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)
        # Pool only spatial dimensions, keep temporal dimension intact
        self.pool = nn.MaxPool3d((1, 2, 2))  # (T, H, W) -> (T, H/2, W/2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x  # pooled, skip


class Up3D(nn.Module):
    """3D upsampling block."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Upsample only spatial dimensions
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, (1, 2, 2), (1, 2, 2))
        self.conv1 = Conv3DBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Frag3D(BranchModule):
    """
    3D U-Net-like segmentation over 3 RGB frames.
    
    Processes 3 consecutive RGB frames to detect debris/smoke/flash masks
    and produce a pooled embedding for downstream fusion.
    
    Input: rgb_seq (B, 3, 3, 480, 640) - 3 frames of RGB
    Outputs:
        - mask_seq (B, 3, 1, 480, 640) - per-frame masks
        - zi (B, 256) - pooled embedding 
        - events (list) - detected events
    """
    
    def __init__(self, base_channels: int = 32, levels: int = 3, 
                 bottleneck_dim: int = 256, norm: str = "group"):
        super().__init__(out_dim=256)
        
        self.levels = levels
        self.base_channels = base_channels
        
        # Initial convolution
        self.input_conv = Conv3DBlock(3, base_channels, norm=norm)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(levels):
            out_ch = base_channels * (2 ** i)
            self.down_blocks.append(Down3D(in_ch, out_ch))
            in_ch = out_ch
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv3DBlock(in_ch, bottleneck_dim, norm=norm),
            Conv3DBlock(bottleneck_dim, bottleneck_dim, norm=norm)
        )
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        in_ch = bottleneck_dim
        for i in range(levels):
            skip_ch = base_channels * (2 ** (levels - 1 - i))
            out_ch = skip_ch
            self.up_blocks.append(Up3D(in_ch, skip_ch, out_ch))
            in_ch = out_ch
            
        # Output head for masks (3 timesteps, 1 channel each)
        self.mask_head = nn.Conv3d(base_channels, 1, 1)
        
        # Global pooling for embedding
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.embedding_proj = nn.Linear(bottleneck_dim, 256)
        
    def forward(self, rgb_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[EventToken]]:
        """
        Forward pass through 3D U-Net.
        
        Args:
            rgb_seq: RGB sequence (B, 3, 3, 480, 640)
            
        Returns:
            tuple: (mask_seq, zi, events)
                - mask_seq: (B, 3, 1, 480, 640) segmentation masks
                - zi: (B, 256) feature embedding
                - events: List of detected events
        """
        B = rgb_seq.shape[0]
        
        # Input processing
        x = self.input_conv(rgb_seq)
        
        # Encoder with skip connections
        skips = []
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global pooling for embedding (from bottleneck)
        pooled = self.global_pool(x).view(B, -1)
        zi = self.embedding_proj(pooled)
        
        # Decoder
        skips = skips[::-1]  # reverse for upsampling
        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[i])
            
        # Generate masks
        mask_logits = self.mask_head(x)  # (B, 1, T, H, W)
        # Reshape to (B, T, 1, H, W) as expected by tasks.yml
        mask_seq = torch.sigmoid(mask_logits.transpose(1, 2))  # (B, T, 1, H, W)
        
        # Generate events based on mask activations
        events = self._extract_events(mask_seq)
        
        return mask_seq, zi, events
    
    def _extract_events(self, mask_seq: torch.Tensor) -> List[EventToken]:
        """Extract events from segmentation masks."""
        events = []
        B, T, C, H, W = mask_seq.shape
        
        for b in range(B):
            for t in range(T):
                mask = mask_seq[b, t, 0]  # (H, W)
                
                # Find significant activations
                threshold = 0.5
                if mask.max() > threshold:
                    # Find center of mass
                    coords = torch.nonzero(mask > threshold, as_tuple=False).float()
                    if len(coords) > 0:
                        center = coords.mean(dim=0)
                        confidence = mask.max().item()
                        
                        event = EventToken(
                            type="debris_detection",
                            value=confidence,
                            t_start=t,
                            t_end=t,
                            quality=confidence,
                            meta={
                                "location": (int(center[1]), int(center[0])),  # (x, y)
                                "mask_area": (mask > threshold).sum().item(),
                                "max_intensity": mask.max().item(),
                                "batch_idx": b
                            }
                        )
                        events.append(event)
        
        return events