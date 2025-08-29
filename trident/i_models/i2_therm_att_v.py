"""
ThermAttentionV: CNN backbone with spatial attention for thermal processing.

Processes temporal RGB sequences with attention-based saliency mapping.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken, FeatureVec


class ThermAttentionV(BranchModule):
    """
    Thermal attention vision module with temporal processing.
    
    Uses ConvNeXt-Tiny or ResNet-18 backbone with spatial attention
    and short temporal convolution over T frames.
    """
    
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        out_dim: int = 256,
        sequence_length: int = 8,
        attention_dim: int = 64,
        pretrained: bool = True
    ) -> None:
        """
        Initialize ThermAttentionV.
        
        Args:
            backbone_name: Timm model name for backbone
            out_dim: Output feature dimension
            sequence_length: Expected temporal sequence length
            attention_dim: Attention module dimension
            pretrained: Use pretrained backbone weights
        """
        super().__init__()
        
        self.out_dim = out_dim
        self.sequence_length = sequence_length
        
        # Backbone CNN (ConvNeXt-Tiny or ResNet-18)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=""   # Remove global pooling
        )
        
        # Get backbone output dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            backbone_features = self.backbone(dummy_input)
        self.backbone_dim = backbone_features.shape[1]
        
        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule(
            in_channels=self.backbone_dim,
            attention_dim=attention_dim
        )
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.backbone_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
        
        # Event detection heads
        self.flash_detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.hotspot_detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(
        self, 
        rgb_roi_t: torch.Tensor
    ) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process temporal RGB sequence and extract features + events.
        
        Args:
            rgb_roi_t: RGB temporal sequence (B, T, C, H, W)
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size, seq_len, channels, height, width = rgb_roi_t.shape
        
        # Reshape for backbone processing
        rgb_flat = rgb_roi_t.view(batch_size * seq_len, channels, height, width)
        
        # Backbone feature extraction
        backbone_features = self.backbone(rgb_flat)  # (B*T, C, H, W)
        feat_h, feat_w = backbone_features.shape[-2:]
        
        # Reshape back to temporal
        backbone_features = backbone_features.view(
            batch_size, seq_len, self.backbone_dim, feat_h, feat_w
        )
        
        # Apply spatial attention to each frame
        attended_features = []
        attention_maps = []
        
        for t in range(seq_len):
            frame_features = backbone_features[:, t]  # (B, C, H, W)
            attended_feat, attention_map = self.spatial_attention(frame_features)
            attended_features.append(attended_feat)
            attention_maps.append(attention_map)
        
        # Stack temporal features
        attended_features = torch.stack(attended_features, dim=1)  # (B, T, C, H, W)
        attention_maps = torch.stack(attention_maps, dim=1)  # (B, T, 1, H, W)
        
        # Global pooling over spatial dimensions
        pooled_features = self.global_pool(
            attended_features.view(batch_size * seq_len, self.backbone_dim, feat_h, feat_w)
        )
        pooled_features = pooled_features.view(batch_size, seq_len, self.backbone_dim)
        
        # Temporal convolution
        temporal_features = self.temporal_conv(
            pooled_features.transpose(1, 2)  # (B, C, T)
        )
        
        # Aggregate temporal features
        aggregated_features = torch.mean(temporal_features, dim=2)  # (B, C)
        
        # Project to output dimension
        projected_features = self.feature_proj(aggregated_features)
        
        # Event detection
        flash_score = self.flash_detector(aggregated_features)
        hotspot_score = self.hotspot_detector(aggregated_features)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Generate event tokens
        events = []
        for b in range(batch_size):
            # Flash detection event
            if flash_score[b] > 0.6:
                # Find peak frame for flash
                frame_intensities = torch.mean(
                    attended_features[b], dim=[1, 2, 3]
                )  # (T,)
                peak_frame = torch.argmax(frame_intensities).item()
                
                events.append(EventToken(
                    type="flash",
                    value=flash_score[b].item(),
                    t_start=peak_frame - 0.5,
                    t_end=peak_frame + 0.5,
                    quality=flash_score[b].item(),
                    meta={
                        "source": "therm_attention_v",
                        "batch_idx": b,
                        "peak_frame": peak_frame
                    }
                ))
            
            # Hotspot detection event
            if hotspot_score[b] > 0.4:
                # Find frames with significant hotspots
                hotspot_frames = []
                for t in range(seq_len):
                    attention_intensity = torch.max(attention_maps[b, t]).item()
                    if attention_intensity > 0.7:
                        hotspot_frames.append(t)
                
                if hotspot_frames:
                    events.append(EventToken(
                        type="hotspot",
                        value=hotspot_score[b].item(),
                        t_start=min(hotspot_frames),
                        t_end=max(hotspot_frames) + 1,
                        quality=hotspot_score[b].item(),
                        meta={
                            "source": "therm_attention_v",
                            "batch_idx": b,
                            "hotspot_frames": hotspot_frames
                        }
                    ))
        
        return feature_vec, events
    
    def get_attention_maps(
        self, 
        rgb_roi_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Get spatial attention maps for visualization.
        
        Args:
            rgb_roi_t: RGB temporal sequence (B, T, C, H, W)
            
        Returns:
            Attention maps (B, T, 1, H, W)
        """
        batch_size, seq_len, channels, height, width = rgb_roi_t.shape
        
        # Reshape for backbone processing
        rgb_flat = rgb_roi_t.view(batch_size * seq_len, channels, height, width)
        
        # Backbone feature extraction
        backbone_features = self.backbone(rgb_flat)
        feat_h, feat_w = backbone_features.shape[-2:]
        
        # Reshape back to temporal
        backbone_features = backbone_features.view(
            batch_size, seq_len, self.backbone_dim, feat_h, feat_w
        )
        
        # Apply spatial attention to each frame
        attention_maps = []
        for t in range(seq_len):
            frame_features = backbone_features[:, t]
            _, attention_map = self.spatial_attention(frame_features)
            attention_maps.append(attention_map)
        
        attention_maps = torch.stack(attention_maps, dim=1)
        
        # Upsample attention maps to input resolution
        if attention_maps.shape[-2:] != (height, width):
            attention_maps = F.interpolate(
                attention_maps.view(batch_size * seq_len, 1, feat_h, feat_w),
                size=(height, width),
                mode="bilinear",
                align_corners=False
            )
            attention_maps = attention_maps.view(batch_size, seq_len, 1, height, width)
        
        return attention_maps


class SpatialAttentionModule(nn.Module):
    """Spatial attention module for saliency mapping."""
    
    def __init__(
        self,
        in_channels: int,
        attention_dim: int = 64,
        reduction_ratio: int = 16
    ) -> None:
        """
        Initialize spatial attention module.
        
        Args:
            in_channels: Input feature channels
            attention_dim: Attention computation dimension
            reduction_ratio: Channel reduction ratio
        """
        super().__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, attention_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, attention_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention to features.
        
        Args:
            features: Input features (B, C, H, W)
            
        Returns:
            Tuple of (attended_features, attention_map)
        """
        # Channel attention
        channel_weights = self.channel_attention(features)
        channel_attended = features * channel_weights
        
        # Spatial attention
        spatial_attention_map = self.spatial_attention(channel_attended)
        spatially_attended = channel_attended * spatial_attention_map
        
        # Feature refinement
        refined_features = self.feature_refine(spatially_attended)
        
        return refined_features, spatial_attention_map


# Default configuration
THERM_ATTENTION_V_CONFIG = {
    "backbone_name": "convnext_tiny",
    "out_dim": 256,
    "sequence_length": 8,
    "attention_dim": 64,
    "pretrained": True
}