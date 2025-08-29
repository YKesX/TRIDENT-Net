"""
DualVisionNet: Siamese encoder for change detection.

Processes pre/post RGB images to detect changes using siamese architecture
with correlation layers and transformer attention.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken, FeatureVec


class DualVisionNet(BranchModule):
    """
    Dual vision network for change detection.
    
    Uses siamese encoder with shared weights, correlation/difference computation,
    and a tiny transformer for change analysis.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet18",
        out_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        correlation_levels: int = 3,
        pretrained: bool = True
    ) -> None:
        """
        Initialize DualVisionNet.
        
        Args:
            encoder_name: Timm model name for siamese encoder
            out_dim: Output feature dimension
            transformer_layers: Number of transformer layers
            transformer_heads: Number of attention heads
            correlation_levels: Number of correlation pyramid levels
            pretrained: Use pretrained encoder weights
        """
        super().__init__()
        
        self.out_dim = out_dim
        self.correlation_levels = correlation_levels
        
        # Siamese encoder (shared weights)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3)  # Multi-scale features
        )
        
        # Get encoder feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            encoder_features = self.encoder(dummy_input)
        self.encoder_dims = [f.shape[1] for f in encoder_features]
        
        # Correlation/difference computation
        self.correlation_processor = CorrelationProcessor(
            feature_dims=self.encoder_dims,
            correlation_levels=correlation_levels
        )
        
        # Transformer for change reasoning
        transformer_dim = sum(self.encoder_dims) * 2  # Correlation + difference features
        self.transformer = TinyTransformer(
            input_dim=transformer_dim,
            hidden_dim=512,
            num_layers=transformer_layers,
            num_heads=transformer_heads
        )
        
        # Change detection heads
        self.change_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.integrity_estimator = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(
        self,
        rgb_pre: torch.Tensor,
        rgb_post: torch.Tensor
    ) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process pre/post images and detect changes.
        
        Args:
            rgb_pre: Pre-event RGB image (B, 3, H, W)
            rgb_post: Post-event RGB image (B, 3, H, W)
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size = rgb_pre.shape[0]
        
        # Siamese feature extraction
        features_pre = self.encoder(rgb_pre)
        features_post = self.encoder(rgb_post)
        
        # Correlation and difference computation
        correlation_features, difference_features = self.correlation_processor(
            features_pre, features_post
        )
        
        # Combine correlation and difference features
        combined_features = torch.cat([correlation_features, difference_features], dim=1)
        
        # Transformer processing
        transformer_output = self.transformer(combined_features)
        
        # Change detection
        change_score = self.change_classifier(transformer_output)
        integrity_score = self.integrity_estimator(transformer_output)
        
        # Feature projection
        projected_features = self.feature_proj(transformer_output)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Generate event tokens
        events = []
        for b in range(batch_size):
            # Change detection event
            if change_score[b] > 0.5:
                events.append(EventToken(
                    type="change_detected",
                    value=change_score[b].item(),
                    t_start=0.0,
                    t_end=1.0,
                    quality=change_score[b].item(),
                    meta={
                        "source": "dual_vision",
                        "batch_idx": b,
                        "integrity_score": integrity_score[b].item()
                    }
                ))
            
            # Integrity assessment event
            if integrity_score[b] < 0.3:  # Low integrity
                events.append(EventToken(
                    type="integrity_loss",
                    value=1.0 - integrity_score[b].item(),
                    t_start=0.0,
                    t_end=1.0,
                    quality=1.0 - integrity_score[b].item(),
                    meta={
                        "source": "dual_vision",
                        "batch_idx": b,
                        "integrity_score": integrity_score[b].item()
                    }
                ))
        
        return feature_vec, events
    
    def get_change_map(
        self,
        rgb_pre: torch.Tensor,
        rgb_post: torch.Tensor
    ) -> torch.Tensor:
        """
        Get spatial change map for visualization.
        
        Args:
            rgb_pre: Pre-event RGB image (B, 3, H, W)
            rgb_post: Post-event RGB image (B, 3, H, W)
            
        Returns:
            Change probability map (B, 1, H, W)
        """
        # Extract features
        features_pre = self.encoder(rgb_pre)
        features_post = self.encoder(rgb_post)
        
        # Use highest resolution features for spatial map
        high_res_pre = features_pre[0]  # Lowest index = highest resolution
        high_res_post = features_post[0]
        
        # Compute difference
        diff = torch.abs(high_res_pre - high_res_post)
        
        # Aggregate across channels
        change_map = torch.mean(diff, dim=1, keepdim=True)
        
        # Normalize to [0, 1]
        change_map = torch.sigmoid(change_map)
        
        # Upsample to input resolution
        target_size = rgb_pre.shape[-2:]
        if change_map.shape[-2:] != target_size:
            change_map = F.interpolate(
                change_map, size=target_size, mode="bilinear", align_corners=False
            )
        
        return change_map


class CorrelationProcessor(nn.Module):
    """Process feature correlations and differences across scales."""
    
    def __init__(
        self,
        feature_dims: List[int],
        correlation_levels: int = 3
    ) -> None:
        """
        Initialize correlation processor.
        
        Args:
            feature_dims: Feature dimensions for each scale
            correlation_levels: Number of correlation levels to compute
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.correlation_levels = correlation_levels
        
        # Projection layers for correlation computation
        self.correlation_projectors = nn.ModuleList()
        for dim in feature_dims:
            self.correlation_projectors.append(
                nn.Conv2d(dim, dim // 2, 1)
            )
        
        # Difference processing layers
        self.difference_processors = nn.ModuleList()
        for dim in feature_dims:
            self.difference_processors.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim, dim // 2, 1)
                )
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(
        self,
        features_pre: List[torch.Tensor],
        features_post: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute correlation and difference features.
        
        Args:
            features_pre: Pre-event features at multiple scales
            features_post: Post-event features at multiple scales
            
        Returns:
            Tuple of (correlation_features, difference_features)
        """
        correlation_outputs = []
        difference_outputs = []
        
        for i, (feat_pre, feat_post) in enumerate(zip(features_pre, features_post)):
            # Project features for correlation
            proj_pre = self.correlation_projectors[i](feat_pre)
            proj_post = self.correlation_projectors[i](feat_post)
            
            # Compute correlation
            correlation = self._compute_correlation(proj_pre, proj_post)
            correlation_pooled = self.global_pool(correlation).squeeze(-1).squeeze(-1)
            correlation_outputs.append(correlation_pooled)
            
            # Compute difference
            difference = torch.abs(feat_pre - feat_post)
            diff_processed = self.difference_processors[i](difference)
            diff_pooled = self.global_pool(diff_processed).squeeze(-1).squeeze(-1)
            difference_outputs.append(diff_pooled)
        
        # Concatenate across scales
        correlation_features = torch.cat(correlation_outputs, dim=1)
        difference_features = torch.cat(difference_outputs, dim=1)
        
        return correlation_features, difference_features
    
    def _compute_correlation(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized cross-correlation between features.
        
        Args:
            feat1: First feature map (B, C, H, W)
            feat2: Second feature map (B, C, H, W)
            
        Returns:
            Correlation map (B, C, H, W)
        """
        # Normalize features
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        
        # Element-wise correlation
        correlation = feat1_norm * feat2_norm
        
        return correlation


class TinyTransformer(nn.Module):
    """Lightweight transformer for change reasoning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize tiny transformer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )
        
        # Positional encoding (simple learned embeddings)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features through transformer.
        
        Args:
            features: Input features (B, C)
            
        Returns:
            Transformed features (B, hidden_dim)
        """
        batch_size = features.shape[0]
        
        # Project to hidden dimension
        x = self.input_proj(features)  # (B, hidden_dim)
        
        # Add positional encoding and reshape for transformer
        x = x.unsqueeze(1) + self.pos_embedding  # (B, 1, hidden_dim)
        
        # Transformer processing
        x = self.transformer(x)  # (B, 1, hidden_dim)
        
        # Extract output
        x = x.squeeze(1)  # (B, hidden_dim)
        x = self.output_norm(x)
        
        return x


# Default configuration
DUAL_VISION_CONFIG = {
    "encoder_name": "resnet18",
    "out_dim": 256,
    "transformer_layers": 2,
    "transformer_heads": 4,
    "correlation_levels": 3,
    "pretrained": True
}