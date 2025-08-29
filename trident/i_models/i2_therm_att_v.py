"""
TRIDENT-I2: Thermal Attention Vision (saliency detection)

Author: Yağızhan Keskin
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken, FeatureVec


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for saliency."""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.conv_query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature map (B, C, H, W)
            
        Returns:
            tuple: (attended_features, attention_map)
        """
        B, C, H, W = x.shape
        
        # Compute queries, keys, values
        query = self.conv_query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C')
        key = self.conv_key(x).view(B, -1, H * W)  # (B, C', H*W)
        value = self.conv_value(x).view(B, -1, H * W)  # (B, C, H*W)
        
        # Attention computation
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = self.softmax(attention)
        
        # Apply attention to values
        attended = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        attended = attended.view(B, C, H, W)
        
        # Residual connection with learnable weight
        output = self.gamma * attended + x
        
        # Generate spatial attention map for visualization
        attention_map = attention.mean(dim=1).view(B, H, W)  # Average over all spatial locations
        
        return output, attention_map


class TemporalConv1D(nn.Module):
    """1D temporal convolution for video processing."""
    
    def __init__(self, channels: int, temporal_kernel: int = 3):
        super().__init__()
        
        self.temporal_conv = nn.Conv1d(
            channels, channels, 
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=channels  # Depthwise convolution
        )
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Temporally processed tensor (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # Reshape for temporal processing
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, H, W, T)
        x = x.view(B * C * H * W, T)  # (B*C*H*W, T)
        
        # Apply 1D convolution across time
        x = self.temporal_conv(x.unsqueeze(1)).squeeze(1)  # (B*C*H*W, T)
        x = self.bn(x.unsqueeze(1)).squeeze(1)
        x = self.relu(x)
        
        # Reshape back
        x = x.view(B, C, H, W, T)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, T, C, H, W)
        
        return x


class ThermAttentionV(BranchModule):
    """
    Thermal Attention Vision module for saliency detection.
    
    Processes temporal RGB sequences to detect flash events and hotspots
    using spatial attention and temporal convolution.
    """
    
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        temporal: str = "1dconv",
        out_dim: int = 256,
        channels: int = 3,
        max_temporal_length: int = 16,
    ):
        super().__init__(out_dim)
        
        self.backbone_name = backbone
        self.temporal_type = temporal
        self.max_temporal_length = max_temporal_length
        
        # Spatial feature extractor
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,  # Remove classifier
            in_chans=channels,
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[1]
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(backbone_dim)
        
        # Temporal processing
        if temporal == "1dconv":
            self.temporal_processor = TemporalConv1D(backbone_dim)
        else:
            raise ValueError(f"Unknown temporal type: {temporal}")
        
        # Feature projection
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim),
        )
        
        # Saliency head
        self.saliency_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        # Event detection heads
        self.flash_detector = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.hotspot_counter = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
    
    def forward(self, rgb_roi_t: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for thermal attention vision.
        
        Args:
            rgb_roi_t: Temporal RGB tensor of shape (B, T, C, H, W)
            
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        B, T, C, H, W = rgb_roi_t.shape
        
        # Limit temporal length if needed
        if T > self.max_temporal_length:
            indices = torch.linspace(0, T-1, self.max_temporal_length, dtype=torch.long)
            rgb_roi_t = rgb_roi_t[:, indices]
            T = self.max_temporal_length
        
        # Process each frame through backbone
        frame_features = []
        spatial_attention_maps = []
        
        for t in range(T):
            frame = rgb_roi_t[:, t]  # (B, C, H, W)
            
            # Extract spatial features
            feat = self.backbone.forward_features(frame)  # (B, D, H', W')
            
            # Apply spatial attention
            attended_feat, attn_map = self.spatial_attention(feat)
            
            frame_features.append(attended_feat)
            spatial_attention_maps.append(attn_map)
        
        # Stack temporal features
        temporal_features = torch.stack(frame_features, dim=1)  # (B, T, D, H', W')
        
        # Apply temporal processing
        temporal_features = self.temporal_processor(temporal_features)
        
        # Use last frame features for saliency and global features
        last_frame_features = temporal_features[:, -1]  # (B, D, H', W')
        
        # Generate saliency map
        saliency = self.saliency_head(last_frame_features)  # (B, 1, H', W')
        
        # Global feature extraction
        global_features = self.feature_head(last_frame_features)  # (B, out_dim)
        
        # Event detection
        flash_prob = self.flash_detector(global_features)
        hotspot_count = self.hotspot_counter(global_features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=global_features)
        
        # Create EventTokens
        events = []
        for b in range(B):
            # Flash detection event
            if flash_prob[b, 0] > 0.7:
                # Compute flash intensity from attention variations
                attention_std = torch.stack(spatial_attention_maps, dim=1)[b].std(dim=0).mean()
                
                events.append(EventToken(
                    type="flash_detected",
                    value=flash_prob[b, 0].item(),
                    t_start=0.0,
                    t_end=T / 30.0,  # Assume 30 FPS
                    quality=flash_prob[b, 0].item(),
                    meta={
                        "attention_variation": attention_std.item(),
                        "saliency_peak": torch.sigmoid(saliency[b]).max().item(),
                        "temporal_frames": T,
                    }
                ))
            
            # Hotspot detection event
            estimated_hotspots = torch.clamp(hotspot_count[b, 0], 0, 10).round().int()
            if estimated_hotspots > 0:
                # Find hotspot locations from saliency map
                saliency_map = torch.sigmoid(saliency[b, 0])
                saliency_thresh = saliency_map.quantile(0.95)
                hotspot_mask = saliency_map > saliency_thresh
                
                events.append(EventToken(
                    type="hotspot_count",
                    value=estimated_hotspots.item(),
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=min(flash_prob[b, 0].item() + 0.2, 1.0),
                    meta={
                        "saliency_threshold": saliency_thresh.item(),
                        "hotspot_area": hotspot_mask.sum().item(),
                        "max_saliency": saliency_map.max().item(),
                    }
                ))
        
        # Store intermediate outputs
        self._last_saliency = saliency
        self._last_attention_maps = spatial_attention_maps
        self._last_temporal_features = temporal_features
        
        return feature_vec, events
    
    def get_saliency_output(self) -> torch.Tensor:
        """Get the last saliency map output."""
        if hasattr(self, '_last_saliency'):
            return self._last_saliency
        else:
            raise RuntimeError("No saliency output available. Run forward() first.")
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """Get spatial attention maps for each frame."""
        if hasattr(self, '_last_attention_maps'):
            return self._last_attention_maps
        else:
            raise RuntimeError("No attention maps available. Run forward() first.")
    
    def get_temporal_features(self) -> torch.Tensor:
        """Get temporal feature maps."""
        if hasattr(self, '_last_temporal_features'):
            return self._last_temporal_features
        else:
            raise RuntimeError("No temporal features available. Run forward() first.")


def create_therm_attention_v(config: dict) -> ThermAttentionV:
    """Factory function to create ThermAttentionV from config."""
    return ThermAttentionV(
        backbone=config.get("backbone", "convnext_tiny"),
        temporal=config.get("temporal", "1dconv"),
        out_dim=config.get("out_dim", 256),
        channels=config.get("channels", 3),
        max_temporal_length=config.get("max_temporal_length", 16),
    )