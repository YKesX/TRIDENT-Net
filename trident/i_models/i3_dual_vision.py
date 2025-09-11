"""
TRIDENT-I3: Dual Vision Network (change detection)

Author: Yağızhan Keskin
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    class timm:
        @staticmethod
        def create_model(model_name, pretrained=True, features_only=False, **kwargs):
            if features_only:
                class FeatureModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
                        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
                    def forward(self, x):
                        return [self.conv1(x), self.conv2(self.conv1(x)), self.conv3(self.conv2(self.conv1(x)))]
                return FeatureModel()
            else:
                return nn.Sequential(nn.Conv2d(3, 64, 7), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1000))

from ..common.types import BranchModule, EventToken, FeatureVec


class CorrelationLayer(nn.Module):
    """Correlation layer for change detection."""
    
    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.max_displacement = max_displacement
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between two feature maps.
        
        Args:
            feat1, feat2: Feature maps of shape (B, C, H, W)
            
        Returns:
            Correlation volume of shape (B, (2*max_displacement+1)^2, H, W)
        """
        B, C, H, W = feat1.shape
        
        # Normalize features
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        correlations = []
        
        for dy in range(-self.max_displacement, self.max_displacement + 1):
            for dx in range(-self.max_displacement, self.max_displacement + 1):
                # Shift feat2 by (dy, dx)
                shifted_feat2 = torch.zeros_like(feat2)
                
                # Handle boundary conditions
                y_start, y_end = max(0, dy), min(H, H + dy)
                x_start, x_end = max(0, dx), min(W, W + dx)
                y2_start, y2_end = max(0, -dy), min(H, H - dy)
                x2_start, x2_end = max(0, -dx), min(W, W - dx)
                
                shifted_feat2[:, :, y_start:y_end, x_start:x_end] = \
                    feat2[:, :, y2_start:y2_end, x2_start:x2_end]
                
                # Compute correlation
                corr = (feat1 * shifted_feat2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
                correlations.append(corr)
        
        correlation_volume = torch.cat(correlations, dim=1)  # (B, D^2, H, W)
        return correlation_volume


class TransformerChangeDetector(nn.Module):
    """Transformer-based change detector."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=3,
        )
        
        self.change_head = nn.Linear(hidden_dim, 1)
        self.integrity_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, correlation_volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process correlation volume with transformer.
        
        Args:
            correlation_volume: Shape (B, C, H, W)
            
        Returns:
            tuple: (change_logits, integrity_score)
        """
        B, C, H, W = correlation_volume.shape
        
        # Flatten spatial dimensions and permute for transformer
        tokens = correlation_volume.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Project to hidden dimension
        tokens = self.input_projection(tokens)  # (B, H*W, hidden_dim)
        
        # Add positional encoding
        seq_len = tokens.shape[1]
        if seq_len <= self.pos_encoding.shape[1]:
            tokens = tokens + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        tokens = self.transformer(tokens)  # (B, H*W, hidden_dim)
        
        # Compute change detection
        change_logits = self.change_head(tokens)  # (B, H*W, 1)
        change_logits = change_logits.view(B, H, W)
        
        # Compute global integrity score
        global_token = tokens.mean(dim=1)  # (B, hidden_dim)
        integrity_score = self.integrity_head(global_token)  # (B, 1)
        
        return change_logits, integrity_score


class DualVisionNet(BranchModule):
    """
    Dual Vision Network for change detection.
    
    Uses siamese encoder with correlation and transformer processing
    to detect changes between pre/post imagery.
    """
    
    def __init__(
        self,
        encoder: str = "efficientnet_b0",
        transformer_heads: int = 4,
        out_dim: int = 256,
        channels: int = 3,
        correlation_max_disp: int = 4,
    ):
        super().__init__(out_dim)
        
        self.encoder_name = encoder
        self.transformer_heads = transformer_heads
        self.correlation_max_disp = correlation_max_disp
        
        # Siamese encoder (shared weights)
        self.encoder = timm.create_model(
            encoder,
            pretrained=True,
            features_only=True,
            in_chans=channels,
        )
        
        # Get encoder output channels
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, 224, 224)
            encoder_features = self.encoder(dummy_input)
            feature_channels = [f.shape[1] for f in encoder_features]
            self.feature_height = encoder_features[-1].shape[2]
            self.feature_width = encoder_features[-1].shape[3]
        
        # Use features from multiple scales
        self.feature_fusion = nn.Conv2d(
            sum(feature_channels[-2:]),  # Use last two feature maps
            256,
            1
        )
        
        # Correlation layer
        self.correlation = CorrelationLayer(max_displacement=correlation_max_disp)
        corr_channels = (2 * correlation_max_disp + 1) ** 2
        
        # Difference branch
        self.diff_processor = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Combine correlation and difference
        combined_channels = corr_channels + 64
        self.transformer_detector = TransformerChangeDetector(
            input_dim=combined_channels,
            hidden_dim=256,
            num_heads=transformer_heads,
        )
        
        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(256 * 2, out_dim * 2),  # 2 images worth of features
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim),
        )
        
        # Event detection heads
        self.change_magnitude_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
        self.change_type_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # appearance, disappearance, modification
            nn.Softmax(dim=1),
        )
    
    def forward(
        self, 
        rgb_pre: torch.Tensor, 
        rgb_post: torch.Tensor
    ) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for change detection.
        
        Args:
            rgb_pre: Pre-event RGB image of shape (B, C, H, W)
            rgb_post: Post-event RGB image of shape (B, C, H, W)
            
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        batch_size = rgb_pre.shape[0]
        
        # Extract features from both images (siamese)
        features_pre = self.encoder(rgb_pre)
        features_post = self.encoder(rgb_post)
        
        # Fuse multi-scale features
        pre_fused = self._fuse_features(features_pre[-2:])
        post_fused = self._fuse_features(features_post[-2:])
        
        # Compute correlation between fused features
        correlation_volume = self.correlation(pre_fused, post_fused)
        
        # Compute simple difference
        diff_features = self.diff_processor(torch.abs(pre_fused - post_fused))
        
        # Combine correlation and difference
        combined_features = torch.cat([correlation_volume, diff_features], dim=1)
        
        # Apply transformer-based change detection
        change_logits, integrity_delta = self.transformer_detector(combined_features)
        
        # Global feature extraction
        pre_global = self.global_pool(pre_fused).flatten(1)
        post_global = self.global_pool(post_fused).flatten(1)
        combined_global = torch.cat([pre_global, post_global], dim=1)
        features = self.feature_head(combined_global)
        
        # Event detection
        change_magnitude = self.change_magnitude_head(features)
        change_type_probs = self.change_type_head(features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=features)
        
        # Create EventTokens
        events = []
        for b in range(batch_size):
            # Change magnitude event
            mag_val = change_magnitude[b, 0].item()
            if abs(mag_val) > 0.2:
                # Determine change type
                type_idx = change_type_probs[b].argmax().item()
                change_types = ["appearance", "disappearance", "modification"]
                change_type = change_types[type_idx]
                
                events.append(EventToken(
                    type=f"change_{change_type}",
                    value=abs(mag_val),
                    t_start=0.0,
                    t_end=1.0,
                    quality=change_type_probs[b, type_idx].item(),
                    meta={
                        "change_magnitude": mag_val,
                        "change_type": change_type,
                        "integrity_delta": integrity_delta[b, 0].item(),
                        "change_area": (torch.sigmoid(change_logits[b]) > 0.5).sum().item(),
                        "max_change_response": torch.sigmoid(change_logits[b]).max().item(),
                    }
                ))
            
            # Integrity assessment event
            integrity_val = integrity_delta[b, 0].item()
            if abs(integrity_val) > 0.3:
                events.append(EventToken(
                    type="integrity_change",
                    value=integrity_val,
                    t_start=0.0,
                    t_end=1.0,
                    quality=min(abs(integrity_val), 1.0),
                    meta={
                        "integrity_direction": "degraded" if integrity_val < 0 else "improved",
                        "integrity_magnitude": abs(integrity_val),
                    }
                ))
        
        # Store intermediate outputs
        self._last_change_mask = change_logits
        self._last_correlation = correlation_volume
        self._last_features = {"pre": features_pre, "post": features_post}
        self._last_integrity_delta = integrity_delta
        
        return feature_vec, events
    
    def _fuse_features(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple scales."""
        # Resize all features to the same spatial size
        target_size = feature_list[-1].shape[-2:]
        
        resized_features = []
        for feat in feature_list:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)
        
        # Concatenate and fuse
        concatenated = torch.cat(resized_features, dim=1)
        fused = self.feature_fusion(concatenated)
        
        return fused
    
    def get_change_output(self) -> torch.Tensor:
        """Get the last change detection output."""
        if hasattr(self, '_last_change_mask'):
            return self._last_change_mask
        else:
            raise RuntimeError("No change output available. Run forward() first.")
    
    def get_correlation_output(self) -> torch.Tensor:
        """Get the correlation volume."""
        if hasattr(self, '_last_correlation'):
            return self._last_correlation
        else:
            raise RuntimeError("No correlation output available. Run forward() first.")
    
    def get_integrity_delta(self) -> torch.Tensor:
        """Get the integrity delta score."""
        if hasattr(self, '_last_integrity_delta'):
            return self._last_integrity_delta
        else:
            raise RuntimeError("No integrity delta available. Run forward() first.")


def create_dual_vision_net(config: dict) -> DualVisionNet:
    """Factory function to create DualVisionNet from config."""
    return DualVisionNet(
        encoder=config.get("encoder", "efficientnet_b0"),
        transformer_heads=config.get("transformer_heads", 4),
        out_dim=config.get("out_dim", 256),
        channels=config.get("channels", 3),
        correlation_max_disp=config.get("correlation_max_disp", 4),
    )