"""
TRIDENT-I2: FlashNetV - Per-frame 2D CNN + temporal convolution for flash dynamics

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken


class FlashNetV(BranchModule):
    """
    Per-frame 2D CNN (shared weights) + shallow temporal conv to capture flash/spark dynamics.
    
    Processes 3 RGB frames with shared 2D backbone, then applies temporal convolution
    to capture flash and spark temporal patterns.
    
    Input: rgb_seq (B, 3, 3, 480, 640) - 3 frames of RGB
    Outputs:
        - saliency_seq (B, 3, 1, 480, 640) - per-frame saliency maps
        - zi (B, 256) - pooled embedding
        - events (list) - detected flash events
    """
    
    def __init__(self, backbone: str = "convnext_tiny", temporal_kernel: int = 3,
                 out_dim: int = 256, dropout: float = 0.1):
        super().__init__(out_dim=out_dim)
        
        self.temporal_kernel = temporal_kernel
        
        # 2D backbone (shared across frames)
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        
        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 480, 640)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
            self.spatial_dims = [(f.shape[2], f.shape[3]) for f in features]
        
        # Feature pyramid fusion
        self.fpn = FeaturePyramidNetwork(self.feature_dims)
        fpn_dim = 256
        
        # Temporal convolution over frame sequence
        self.temporal_conv = nn.Conv1d(fpn_dim, fpn_dim, temporal_kernel, padding=temporal_kernel//2)
        self.temporal_norm = nn.BatchNorm1d(fpn_dim)
        
        # Saliency head
        self.saliency_head = nn.Sequential(
            nn.Conv2d(fpn_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Global pooling and embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_proj = nn.Sequential(
            nn.Linear(fpn_dim, out_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, rgb_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[EventToken]]:
        """
        Forward pass through FlashNetV.
        
        Args:
            rgb_seq: RGB sequence (B, 3, 3, 480, 640)
            
        Returns:
            tuple: (saliency_seq, zi, events)
                - saliency_seq: (B, 3, 1, 480, 640) saliency maps
                - zi: (B, 256) feature embedding
                - events: List of detected flash events
        """
        B, T, C, H, W = rgb_seq.shape
        
        # Process each frame through shared backbone
        frame_features = []
        for t in range(T):
            frame = rgb_seq[:, t]  # (B, 3, H, W)
            features = self.backbone(frame)
            fused = self.fpn(features)  # (B, 256, H', W')
            frame_features.append(fused)
        
        # Stack temporal features (B, T, C, H', W')
        temporal_features = torch.stack(frame_features, dim=1)
        B, T, C, H_feat, W_feat = temporal_features.shape
        
        # Apply temporal convolution
        # Reshape for 1D conv: (B*H'*W', C, T)
        temp_reshaped = temporal_features.permute(0, 3, 4, 2, 1).contiguous()
        temp_reshaped = temp_reshaped.view(B * H_feat * W_feat, C, T)
        
        # Temporal conv
        temp_conv = self.temporal_conv(temp_reshaped)
        temp_conv = self.temporal_norm(temp_conv)
        temp_conv = F.relu(temp_conv, inplace=True)
        
        # Reshape back: (B, H', W', C, T) -> (B, T, C, H', W')
        temp_conv = temp_conv.view(B, H_feat, W_feat, C, T)
        temp_conv = temp_conv.permute(0, 4, 3, 1, 2).contiguous()
        
        # Generate saliency maps for each frame
        saliency_seq = []
        embeddings = []
        
        for t in range(T):
            feat_t = temp_conv[:, t]  # (B, C, H', W')
            
            # Upsample to original resolution if needed
            if (H_feat, W_feat) != (H, W):
                feat_t = F.interpolate(feat_t, size=(H, W), mode='bilinear', align_corners=False)
            
            # Generate saliency
            saliency = self.saliency_head(feat_t)  # (B, 1, H, W)
            saliency_seq.append(saliency)
            
            # Global pooling for embedding
            pooled = self.global_pool(feat_t).view(B, -1)  # (B, C)
            embeddings.append(pooled)
        
        # Stack saliency maps
        saliency_seq = torch.stack(saliency_seq, dim=1)  # (B, T, 1, H, W)
        
        # Average embeddings across time and project
        avg_embedding = torch.stack(embeddings, dim=1).mean(dim=1)  # (B, C)
        zi = self.embedding_proj(avg_embedding)  # (B, out_dim)
        
        # Extract flash events
        events = self._extract_flash_events(saliency_seq)
        
        return saliency_seq, zi, events
    
    def _extract_flash_events(self, saliency_seq: torch.Tensor) -> List[EventToken]:
        """Extract flash events from saliency sequence."""
        events = []
        B, T, C, H, W = saliency_seq.shape
        
        flash_threshold = 0.7
        
        for b in range(B):
            for t in range(T):
                saliency = saliency_seq[b, t, 0]  # (H, W)
                
                # Find flash peaks
                max_val = saliency.max().item()
                if max_val > flash_threshold:
                    # Find peak location
                    peak_coords = torch.nonzero(saliency == saliency.max(), as_tuple=False)[0]
                    peak_x, peak_y = int(peak_coords[1]), int(peak_coords[0])
                    
                    # Check for temporal persistence (flash vs noise)
                    temporal_score = self._compute_temporal_flash_score(saliency_seq[b], t, peak_x, peak_y)
                    
                    if temporal_score > 0.3:  # Minimum temporal consistency
                        event = EventToken(
                            event_type="flash_detection",
                            confidence=max_val * temporal_score,
                            location=(peak_x, peak_y),
                            timestamp=t,
                            source="flashnetv",
                            metadata={
                                "intensity": max_val,
                                "temporal_score": temporal_score,
                                "batch_idx": b
                            }
                        )
                        events.append(event)
        
        return events
    
    def _compute_temporal_flash_score(self, saliency_seq: torch.Tensor, t: int, x: int, y: int) -> float:
        """Compute temporal consistency score for flash detection."""
        T = saliency_seq.shape[0]
        window = 1  # Check ±1 frame
        
        scores = []
        center_val = saliency_seq[t, 0, y, x].item()
        
        for dt in range(-window, window + 1):
            frame_idx = t + dt
            if 0 <= frame_idx < T and frame_idx != t:
                neighbor_val = saliency_seq[frame_idx, 0, y, x].item()
                # Flash should have temporal gradient (rise/fall)
                if dt < 0:  # Before flash
                    scores.append(max(0, center_val - neighbor_val))
                else:  # After flash
                    scores.append(max(0, center_val - neighbor_val))
        
        return sum(scores) / len(scores) if scores else 0.0


class FeaturePyramidNetwork(nn.Module):
    """Simple FPN for multi-scale feature fusion."""
    
    def __init__(self, feature_dims: List[int], out_dim: int = 256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, out_dim, 1) for dim in feature_dims
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, 3, padding=1) for _ in feature_dims
        ])
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features."""
        # Process features from deepest to shallowest
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample and add
            upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], 
                                    mode='bilinear', align_corners=False)
            laterals[i] = laterals[i] + upsampled
        
        # Apply FPN convs
        fpn_features = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        # Return the finest resolution feature
        return fpn_features[0]