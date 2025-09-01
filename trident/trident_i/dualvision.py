"""
TRIDENT-I3: DualVision - Siamese encoder for structural change detection

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken


class DualVision(BranchModule):
    """
    Siamese encoder on pre/post RGB frames for structural change detection.
    
    Uses siamese architecture to encode pre and post frames, then applies
    correlation and transformer to detect structural changes.
    
    Input: 
        - rgb_pre (B, 3, 480, 640) - pre-event frame
        - rgb_post (B, 3, 480, 640) - post-event frame
    Outputs:
        - change_mask (B, 1, 480, 640) - change detection mask
        - integrity_delta (B, 1) - structural integrity change score
        - zi (B, 256) - feature embedding
        - events (list) - detected change events
    """
    
    def __init__(self, encoder: str = "efficientnet_b0", transformer_heads: int = 4,
                 transformer_layers: int = 2, out_dim: int = 256):
        super().__init__(out_dim=out_dim)
        
        # Siamese encoder (shared weights)
        self.encoder = timm.create_model(encoder, pretrained=True, features_only=True)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 480, 640)
            features = self.encoder(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
            # Use the deepest feature for correlation
            self.feat_dim = self.feature_dims[-1]
            self.feat_spatial = features[-1].shape[2:]
        
        # Cross-correlation module
        self.correlation = CrossCorrelation(self.feat_dim)
        
        # Change detection transformer
        self.change_transformer = ChangeTransformer(
            d_model=self.feat_dim,
            n_heads=transformer_heads,
            n_layers=transformer_layers
        )
        
        # Change mask decoder
        self.mask_decoder = ChangeDecoder(self.feat_dim)
        
        # Integrity assessment head
        self.integrity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Feature embedding projection
        self.embedding_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, rgb_pre: torch.Tensor, rgb_post: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[EventToken]]:
        """
        Forward pass through DualVision.
        
        Args:
            rgb_pre: Pre-event RGB frame (B, 3, 480, 640)
            rgb_post: Post-event RGB frame (B, 3, 480, 640)
            
        Returns:
            tuple: (change_mask, integrity_delta, zi, events)
                - change_mask: (B, 1, 480, 640) change detection mask
                - integrity_delta: (B, 1) structural integrity change
                - zi: (B, 256) feature embedding
                - events: List of detected change events
        """
        B = rgb_pre.shape[0]
        
        # Extract features from both frames (siamese)
        pre_features = self.encoder(rgb_pre)
        post_features = self.encoder(rgb_post)
        
        # Use deepest features for change analysis
        pre_feat = pre_features[-1]  # (B, C, H', W')
        post_feat = post_features[-1]
        
        # Compute cross-correlation
        correlation_map = self.correlation(pre_feat, post_feat)  # (B, C, H', W')
        
        # Apply change transformer
        change_features = self.change_transformer(pre_feat, post_feat, correlation_map)
        
        # Generate change mask
        change_mask = self.mask_decoder(change_features)  # (B, 1, H, W)
        
        # Compute integrity delta
        integrity_delta = self.integrity_head(change_features)  # (B, 1)
        
        # Generate embedding
        zi = self.embedding_proj(change_features)  # (B, out_dim)
        
        # Extract change events
        events = self._extract_change_events(change_mask, integrity_delta)
        
        return change_mask, integrity_delta, zi, events
    
    def _extract_change_events(self, change_mask: torch.Tensor, integrity_delta: torch.Tensor) -> List[EventToken]:
        """Extract change events from detection outputs."""
        events = []
        B, _, H, W = change_mask.shape
        
        change_threshold = 0.6
        integrity_threshold = 0.3
        
        for b in range(B):
            mask = change_mask[b, 0]  # (H, W)
            integrity = integrity_delta[b, 0].item()
            
            # Find significant change regions
            if mask.max() > change_threshold or abs(integrity) > integrity_threshold:
                # Find change centroid
                significant_pixels = mask > change_threshold
                if significant_pixels.sum() > 0:
                    coords = torch.nonzero(significant_pixels, as_tuple=False).float()
                    centroid = coords.mean(dim=0)
                    
                    # Determine change type based on integrity delta
                    if integrity < -integrity_threshold:
                        change_type = "structural_damage"
                    elif integrity > integrity_threshold:
                        change_type = "structural_addition"
                    else:
                        change_type = "surface_change"
                    
                    event = EventToken(
                        event_type=change_type,
                        confidence=min(mask.max().item(), abs(integrity)),
                        location=(int(centroid[1]), int(centroid[0])),  # (x, y)
                        timestamp=1,  # Post-event
                        source="dualvision",
                        metadata={
                            "integrity_delta": integrity,
                            "change_area": significant_pixels.sum().item(),
                            "max_change": mask.max().item(),
                            "batch_idx": b
                        }
                    )
                    events.append(event)
        
        return events


class CrossCorrelation(nn.Module):
    """Cross-correlation module for change detection."""
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        
        # Projection layers for correlation
        self.pre_proj = nn.Conv2d(feat_dim, feat_dim, 1)
        self.post_proj = nn.Conv2d(feat_dim, feat_dim, 1)
        self.corr_proj = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
        
    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        """Compute cross-correlation between pre and post features."""
        # Project features
        pre_proj = self.pre_proj(pre_feat)
        post_proj = self.post_proj(post_feat)
        
        # Compute element-wise correlation
        correlation = pre_proj * post_proj
        
        # Apply convolution to capture spatial relationships
        correlation = self.corr_proj(correlation)
        
        return correlation


class ChangeTransformer(nn.Module):
    """Transformer for change feature refinement."""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Conv2d(d_model * 3, d_model, 1)  # pre + post + corr
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_proj = nn.Conv2d(d_model, d_model, 3, padding=1)
        
    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor, 
                correlation: torch.Tensor) -> torch.Tensor:
        """Apply transformer to change features."""
        B, C, H, W = pre_feat.shape
        
        # Concatenate features
        combined = torch.cat([pre_feat, post_feat, correlation], dim=1)  # (B, 3*C, H, W)
        
        # Project to model dimension
        x = self.input_proj(combined)  # (B, C, H, W)
        
        # Reshape for transformer (B, H*W, C)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Apply transformer
        x_trans = self.transformer(x_flat)  # (B, H*W, C)
        
        # Reshape back to spatial
        x_spatial = x_trans.permute(0, 2, 1).view(B, C, H, W)
        
        # Final projection
        output = self.output_proj(x_spatial)
        
        return output


class ChangeDecoder(nn.Module):
    """Decoder for change mask generation."""
    
    def __init__(self, feat_dim: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(feat_dim // 2, feat_dim // 4, 3, padding=1),
            nn.BatchNorm2d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(feat_dim // 4, feat_dim // 8, 3, padding=1),
            nn.BatchNorm2d(feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(feat_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to change mask."""
        return self.decoder(x)