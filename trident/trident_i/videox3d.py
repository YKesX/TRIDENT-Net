"""
Variable-T 3D U-Net for RGB video fragment detection in TRIDENT-Net.

Processes RGB clips with temporal pooling and fragment mask generation.

Author: Yağızhan Keskin
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import EventToken


class VideoFrag3Dv2(nn.Module):
    """
    Variable-T 3D U-Net-lite for RGB clip processing.
    
    Processes RGB video clips of variable temporal length T and generates:
    - Per-frame saliency/fragment masks
    - Global embedding vector
    - Event tokens from peak mask trajectories
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 3,
        temporal_kernel: int = 3,
        temporal_stride: int = 2,
        norm: str = "group",
        act: str = "gelu",
        out_embed_dim: int = 512
    ):
        """
        Initialize VideoFrag3Dv2.
        
        Args:
            in_channels: Input channels (3 for RGB)
            base_channels: Base number of channels for conv layers
            depth: Number of encoder/decoder levels
            temporal_kernel: Temporal convolution kernel size
            temporal_stride: Temporal stride for downsampling
            norm: Normalization type ("group", "batch", "instance")
            act: Activation function ("gelu", "relu", "swish")
            out_embed_dim: Output embedding dimension (zi)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        self.out_embed_dim = out_embed_dim
        
        # Build encoder layers
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pool_channels = []  # Store channel info for dynamic pool creation
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            
            # Encoder block
            block = self._make_conv_block(
                in_ch, out_ch, temporal_kernel, norm, act
            )
            self.encoder_blocks.append(block)
            
            # Store pooling info (except last layer)
            if i < depth - 1:
                self.encoder_pool_channels.append(out_ch)
            
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** (depth - 1))
        self.bottleneck = self._make_conv_block(
            bottleneck_ch, bottleneck_ch * 2, temporal_kernel, norm, act
        )
        
        # Global pooling for embedding
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.embedding_proj = nn.Linear(bottleneck_ch * 2, out_embed_dim)
        
        # Build decoder layers
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample_channels = []  # Store channel info for dynamic upsample creation
        
        # Track encoder channel dimensions for skip connections
        encoder_channels = []
        ch = base_channels
        for i in range(depth):
            encoder_channels.append(ch)
            ch *= 2
        encoder_channels.reverse()  # Reverse for decoder order
        
        in_ch = bottleneck_ch * 2
        for i in range(depth):
            out_ch = encoder_channels[i]
            
            # Store upsample info (except first decoder layer)
            if i > 0:
                self.decoder_upsample_channels.append(in_ch)
                in_ch = in_ch // 2
            
            # Decoder block (with skip connection handling)
            skip_ch = encoder_channels[i] if i < len(encoder_channels) else 0
            block = self._make_conv_block(
                in_ch + skip_ch, out_ch, temporal_kernel, norm, act
            )
            self.decoder_blocks.append(block)
            
            in_ch = out_ch
        
        # Final output layers
        self.final_conv = nn.Conv3d(
            base_channels, 1,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0)
        )
        
        # Event detection parameters
        self.event_threshold = 0.5
        
    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int,
        norm: str,
        act: str
    ) -> nn.Module:
        """Create a 3D convolution block with normalization and activation."""
        layers = []
        
        # Main convolution
        conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 3, 3),
            padding=(temporal_kernel // 2, 1, 1)
        )
        layers.append(conv)
        
        # Normalization
        if norm == "group":
            num_groups = min(8, out_channels)  # Ensure valid group count
            layers.append(nn.GroupNorm(num_groups, out_channels))
        elif norm == "batch":
            layers.append(nn.BatchNorm3d(out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm3d(out_channels))
        
        # Activation
        if act == "gelu":
            layers.append(nn.GELU())
        elif act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "swish":
            layers.append(nn.SiLU())
        
        return nn.Sequential(*layers)
    
    def _create_adaptive_pool(self, channels: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """Create adaptive pooling layer that adjusts kernel size based on input dimensions."""
        T, H, W = input_shape
        
        # Determine temporal kernel/stride based on input temporal size
        temporal_kernel = min(self.temporal_stride, T)
        temporal_stride = min(self.temporal_stride, T)
        
        # For very small temporal dimensions, use 1x1 temporal kernel
        if T == 1:
            temporal_kernel = 1
            temporal_stride = 1
        
        pool = nn.Conv3d(
            channels, channels,
            kernel_size=(temporal_kernel, 2, 2),
            stride=(temporal_stride, 2, 2),
            padding=(0, 1, 1)
        )
        
        # Move to same device as the model
        if hasattr(self, 'encoder_blocks') and len(self.encoder_blocks) > 0:
            device = next(self.encoder_blocks[0].parameters()).device
            pool = pool.to(device)
        
        return pool
    
    def _create_adaptive_upsample(self, in_channels: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """Create adaptive upsampling layer that adjusts kernel size based on input dimensions."""
        T, H, W = input_shape
        
        # Determine temporal kernel/stride based on input temporal size
        temporal_kernel = min(self.temporal_stride, T * self.temporal_stride)
        temporal_stride = min(self.temporal_stride, T * self.temporal_stride)
        
        # For very small temporal dimensions, use 1x1 temporal kernel
        if T == 1:
            temporal_kernel = 1
            temporal_stride = 1
        
        upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2,
            kernel_size=(temporal_kernel, 2, 2),
            stride=(temporal_stride, 2, 2),
            padding=(0, 0, 0)
        )
        
        # Move to same device as the model
        if hasattr(self, 'encoder_blocks') and len(self.encoder_blocks) > 0:
            device = next(self.encoder_blocks[0].parameters()).device
            upsample = upsample.to(device)
        
        return upsample
    
    def forward(self, rgb: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through VideoFrag3Dv2.
        
        Args:
            rgb: RGB video tensor [B, 3, T, H, W]
            
        Returns:
            Dictionary containing:
                - mask_seq: Fragment masks [B, T, 1, H, W]
                - zi: Global embedding [B, 512]
                - events: List of EventToken objects
        """
        B, C, T, H, W = rgb.shape
        
        # Store encoder features for skip connections
        encoder_features = []
        x = rgb
        
        # Encoder path
        for i, block in enumerate(self.encoder_blocks[:-1]):
            x = block(x)
            encoder_features.append(x)
            
            # Apply adaptive pooling
            pool_channels = self.encoder_pool_channels[i]
            current_shape = x.shape[2:]  # T, H, W
            pool = self._create_adaptive_pool(pool_channels, current_shape)
            x = pool(x)
        
        # Final encoder block (no pooling)
        x = self.encoder_blocks[-1](x)
        encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Global embedding extraction
        pooled = self.global_pool(x)  # [B, C, 1, 1, 1]
        pooled = pooled.view(B, -1)  # [B, C]
        zi = self.embedding_proj(pooled)  # [B, out_embed_dim]
        
        # Decoder path
        skip_features = encoder_features[::-1]  # Reverse order for skip connections
        
        for i, block in enumerate(self.decoder_blocks):
            # Upsample (except first decoder layer)
            if i > 0 and i - 1 < len(self.decoder_upsample_channels):
                upsample_channels = self.decoder_upsample_channels[i - 1]
                current_shape = x.shape[2:]  # T, H, W
                upsample = self._create_adaptive_upsample(upsample_channels, current_shape)
                x = upsample(x)
            
            # Add skip connection if available and shapes match
            if i < len(skip_features):
                skip = skip_features[i]
                
                # Adjust skip feature spatial dimensions to match current x
                if skip.shape != x.shape:
                    # Resize skip to match x spatial dimensions
                    _, _, skip_T, skip_H, skip_W = skip.shape
                    _, _, x_T, x_H, x_W = x.shape
                    
                    if skip_T != x_T or skip_H != x_H or skip_W != x_W:
                        skip = F.interpolate(
                            skip,
                            size=(x_T, x_H, x_W),
                            mode='trilinear',
                            align_corners=False
                        )
                
                # Concatenate along channel dimension
                if skip.shape[2:] == x.shape[2:]:  # Same spatial dimensions
                    x = torch.cat([x, skip], dim=1)
            
            x = block(x)
        
        # Final convolution to get masks
        mask_logits = self.final_conv(x)  # [B, 1, T_out, H_out, W_out]
        
        # Upsample to original temporal and spatial resolution
        mask_logits = F.interpolate(
            mask_logits,
            size=(T, H, W),
            mode='trilinear',
            align_corners=False
        )
        
        # Apply sigmoid to get masks in [0, 1]
        mask_seq = torch.sigmoid(mask_logits)  # [B, 1, T, H, W]
        
        # Rearrange to match expected output format [B, T, 1, H, W]
        mask_seq = mask_seq.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
        
        # Extract events from mask trajectories
        events = self._extract_events(mask_seq, threshold=self.event_threshold)
        
        return {
            'mask_seq': mask_seq,
            'zi': zi,
            'events': events
        }
    
    def _extract_events(
        self,
        mask_seq: torch.Tensor,
        threshold: float = 0.5
    ) -> List[EventToken]:
        """
        Extract event tokens from mask sequence trajectories.
        
        Args:
            mask_seq: Mask sequence [B, T, 1, H, W]
            threshold: Minimum score threshold for events
            
        Returns:
            List of EventToken objects
        """
        B, T, _, H, W = mask_seq.shape
        events = []
        
        for b in range(B):
            batch_masks = mask_seq[b]  # [T, 1, H, W]
            
            # Compute spatial max pooling to get frame-level scores
            frame_scores = F.max_pool2d(
                batch_masks.squeeze(1),  # [T, H, W]
                kernel_size=(H // 8, W // 8),  # Pool to reduce spatial dimension
                stride=(H // 8, W // 8)
            ).max(dim=-1)[0].max(dim=-1)[0]  # [T]
            
            # Find peaks in temporal domain
            peak_indices = self._find_peaks(frame_scores, threshold)
            
            for peak_idx in peak_indices:
                score = float(frame_scores[peak_idx].item())
                
                # Convert frame index to milliseconds (assuming 24 FPS)
                t_ms = int(peak_idx * (1000 / 24))
                
                event = EventToken(
                    type="rgb_activity",
                    score=score,
                    t_ms=t_ms,
                    meta={
                        'frame_idx': int(peak_idx),
                        'spatial_max': score,
                        'batch_idx': b
                    }
                )
                events.append(event)
        
        return events
    
    def _find_peaks(
        self,
        signal: torch.Tensor,
        threshold: float,
        min_distance: int = 3
    ) -> List[int]:
        """
        Find peak indices in 1D signal.
        
        Args:
            signal: 1D tensor [T]
            threshold: Minimum peak value
            min_distance: Minimum distance between peaks
            
        Returns:
            List of peak indices
        """
        signal_np = signal.detach().cpu().numpy()
        peaks = []
        
        for i in range(1, len(signal_np) - 1):
            # Check if local maximum
            if (signal_np[i] > signal_np[i - 1] and 
                signal_np[i] > signal_np[i + 1] and
                signal_np[i] >= threshold):
                
                # Check minimum distance constraint
                valid_peak = True
                for existing_peak in peaks:
                    if abs(i - existing_peak) < min_distance:
                        # Keep the higher peak
                        if signal_np[i] > signal_np[existing_peak]:
                            peaks.remove(existing_peak)
                        else:
                            valid_peak = False
                        break
                
                if valid_peak:
                    peaks.append(i)
        
        return sorted(peaks)
    
    def get_output_shapes(self, input_shape: Tuple[int, ...]) -> Dict[str, Tuple[int, ...]]:
        """
        Get expected output shapes for given input shape.
        
        Args:
            input_shape: Input tensor shape (B, C, T, H, W)
            
        Returns:
            Dictionary of output shapes
        """
        B, C, T, H, W = input_shape
        
        return {
            'mask_seq': (B, T, 1, H, W),
            'zi': (B, self.out_embed_dim),
            'events': "List[EventToken]"
        }