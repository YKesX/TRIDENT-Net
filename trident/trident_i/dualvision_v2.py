"""
Siamese pre/post change detector with transformer for TRIDENT-Net.

Analyzes change between representative frames around hit events.

Author: Yağızhan Keskin
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

from ..common.types import EventToken


class DualVisionV2(nn.Module):
    """
    Siamese pre/post change detection with tiny transformer.
    
    Selects representative pre/post frames around hit timing,
    computes change via cross-correlation and transformer processing.
    """
    
    def __init__(
        self,
        encoder_backbone: str = "efficientnet_b0",
        transformer: Dict[str, Any] = None,
        out_embed_dim: int = 256,
        frame_picker: str = "around_hit"
    ):
        """
        Initialize DualVisionV2.
        
        Args:
            encoder_backbone: Backbone model name (e.g., "efficientnet_b0")
            transformer: Transformer configuration dict
            out_embed_dim: Output embedding dimension
            frame_picker: Frame selection strategy
        """
        super().__init__()
        
        self.encoder_backbone = encoder_backbone
        self.out_embed_dim = out_embed_dim
        self.frame_picker = frame_picker
        
        # Transformer config defaults
        if transformer is None:
            transformer = {}
        self.transformer_cfg = {
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 2,
            'mlp': 256,
            'dropout': 0.1,
            **transformer
        }
        
        # Build encoder backbone
        self.encoder = self._build_encoder(encoder_backbone)
        self.encoder_dim = self._get_encoder_dim()
        
        # Feature projection to transformer dimension
        self.feature_proj = nn.Linear(self.encoder_dim, self.transformer_cfg['d_model'])
        
        # Tiny transformer for cross-correlation processing
        self.transformer = self._build_transformer()
        
        # Change detection heads
        self.change_decoder = self._build_change_decoder()
        
        # Final projections
        self.integrity_head = nn.Sequential(
            nn.Linear(self.transformer_cfg['d_model'], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.embedding_head = nn.Linear(self.transformer_cfg['d_model'], out_embed_dim)
        
    def _build_encoder(self, backbone_name: str) -> nn.Module:
        """Build feature encoder backbone."""
        if timm is None:
            # Fallback to simple CNN if timm not available
            return self._build_simple_cnn()
        
        try:
            # Load pretrained backbone
            model = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                global_pool='',
                out_indices=[-1]  # Last feature map
            )
            return model
        except Exception:
            # Fallback if model not found
            return self._build_simple_cnn()
    
    def _build_simple_cnn(self) -> nn.Module:
        """Build simple CNN encoder as fallback."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 14))  # Fixed spatial size
        )
    
    def _get_encoder_dim(self) -> int:
        """Get encoder output dimension."""
        # Test forward pass to determine dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            try:
                if hasattr(self.encoder, 'forward_features'):
                    features = self.encoder.forward_features(dummy_input)
                else:
                    features = self.encoder(dummy_input)
                
                if isinstance(features, (list, tuple)):
                    features = features[-1]
                
                return features.shape[1]  # Channel dimension
            except Exception:
                return 256  # Default fallback
    
    def _build_transformer(self) -> nn.Module:
        """Build tiny transformer for cross-correlation processing."""
        cfg = self.transformer_cfg
        
        # Single transformer layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=cfg['d_model'],
            nhead=cfg['n_heads'],
            dim_feedforward=cfg['mlp'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True
        )
        
        transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=cfg['n_layers']
        )
        
        return transformer
    
    def _build_change_decoder(self) -> nn.Module:
        """Build decoder for change mask generation."""
        d_model = self.transformer_cfg['d_model']
        
        return nn.Sequential(
            # Upsample to intermediate resolution
            nn.ConvTranspose2d(d_model, 128, 4, 2, 1),  # 2x upsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Further upsample
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 2x upsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # More upsampling
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 2x upsample
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        times_ms: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through DualVisionV2.
        
        Args:
            rgb: RGB video tensor [B, 3, T, H, W]
            times_ms: Optional timing information for frame selection
            
        Returns:
            Dictionary containing:
                - change_mask: Change mask [B, 1, H, W]
                - integrity_delta: Integrity change [-1, 1] [B, 1]
                - zi: Embedding vector [B, 256]
                - events: List of EventToken objects
        """
        B, C, T, H, W = rgb.shape
        
        # Select representative pre/post frames
        pre_frame, post_frame = self._select_frames(rgb, times_ms)
        
        # Extract features from both frames (Siamese)
        pre_features = self._encode_frame(pre_frame)  # [B, D, H_f, W_f]
        post_features = self._encode_frame(post_frame)  # [B, D, H_f, W_f]
        
        # Project features to transformer dimension
        B_f, D_f, H_f, W_f = pre_features.shape
        
        # Reshape for transformer processing
        pre_tokens = pre_features.view(B_f, D_f, -1).permute(0, 2, 1)  # [B, H_f*W_f, D_f]
        post_tokens = post_features.view(B_f, D_f, -1).permute(0, 2, 1)  # [B, H_f*W_f, D_f]
        
        pre_tokens = self.feature_proj(pre_tokens)  # [B, H_f*W_f, d_model]
        post_tokens = self.feature_proj(post_tokens)  # [B, H_f*W_f, d_model]
        
        # Concatenate pre/post tokens for cross-attention
        combined_tokens = torch.cat([pre_tokens, post_tokens], dim=1)  # [B, 2*H_f*W_f, d_model]
        
        # Add positional encoding
        combined_tokens = self._add_positional_encoding(combined_tokens)
        
        # Process through transformer
        processed_tokens = self.transformer(combined_tokens)  # [B, 2*H_f*W_f, d_model]
        
        # Split back to pre/post
        n_spatial = H_f * W_f
        pre_processed = processed_tokens[:, :n_spatial]  # [B, H_f*W_f, d_model]
        post_processed = processed_tokens[:, n_spatial:]  # [B, H_f*W_f, d_model]
        
        # Compute change features
        change_features = post_processed - pre_processed  # [B, H_f*W_f, d_model]
        
        # Reshape back to spatial format for decoder
        change_spatial = change_features.permute(0, 2, 1).view(B_f, -1, H_f, W_f)  # [B, d_model, H_f, W_f]
        
        # Generate change mask
        change_mask = self.change_decoder(change_spatial)  # [B, 1, H_decoded, W_decoded]
        
        # Upsample change mask to input resolution
        change_mask = F.interpolate(
            change_mask,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # Compute global integrity delta
        global_change = change_features.mean(dim=1)  # [B, d_model]
        integrity_delta = self.integrity_head(global_change)  # [B, 1]
        
        # Compute embedding
        global_features = processed_tokens.mean(dim=1)  # [B, d_model]
        zi = self.embedding_head(global_features)  # [B, out_embed_dim]
        
        # Extract events
        events = self._extract_events(change_mask, integrity_delta)
        
        return {
            'change_mask': change_mask,
            'integrity_delta': integrity_delta,
            'zi': zi,
            'events': events
        }
    
    def _select_frames(
        self,
        rgb: torch.Tensor,
        times_ms: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select representative pre/post frames.
        
        Args:
            rgb: RGB video [B, 3, T, H, W]
            times_ms: Timing information
            
        Returns:
            tuple: (pre_frame, post_frame) each [B, 3, H, W]
        """
        B, C, T, H, W = rgb.shape
        
        if times_ms and 'hit_ms' in times_ms and times_ms['hit_ms'] is not None:
            # Use timing information to select frames
            hit_ms = times_ms['hit_ms']
            shoot_ms = times_ms.get('shoot_ms', hit_ms)
            
            # Estimate frame indices (assuming 24 FPS)
            fps = 24
            
            # Select frames around shoot and hit
            shoot_frame_idx = max(0, min(T - 1, int(shoot_ms * fps / 1000) if shoot_ms else T // 3))
            hit_frame_idx = max(0, min(T - 1, int(hit_ms * fps / 1000) if hit_ms else 2 * T // 3))
            
            # Add small offset to avoid identical frames
            pre_frame_idx = max(0, shoot_frame_idx - 2)
            post_frame_idx = min(T - 1, hit_frame_idx + 2)
        else:
            # Default frame selection: early and late in sequence
            pre_frame_idx = T // 4
            post_frame_idx = 3 * T // 4
        
        # Extract frames
        pre_frame = rgb[:, :, pre_frame_idx]  # [B, 3, H, W]
        post_frame = rgb[:, :, post_frame_idx]  # [B, 3, H, W]
        
        return pre_frame, post_frame
    
    def _encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode single frame through backbone.
        
        Args:
            frame: Single frame [B, 3, H, W]
            
        Returns:
            Feature tensor [B, D, H_f, W_f]
        """
        if hasattr(self.encoder, 'forward_features'):
            # timm model
            features = self.encoder.forward_features(frame)
            if isinstance(features, (list, tuple)):
                features = features[-1]
        else:
            # Simple CNN
            features = self.encoder(frame)
        
        return features
    
    def _add_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token sequence.
        
        Args:
            tokens: Token sequence [B, N, D]
            
        Returns:
            Tokens with positional encoding [B, N, D]
        """
        B, N, D = tokens.shape
        
        # Simple learned positional encoding
        if not hasattr(self, 'pos_encoding'):
            self.pos_encoding = nn.Parameter(torch.randn(N, D) * 0.02)
        
        # Broadcast positional encoding
        pos_enc = self.pos_encoding[:N].unsqueeze(0).expand(B, -1, -1)
        
        return tokens + pos_enc
    
    def _extract_events(
        self,
        change_mask: torch.Tensor,
        integrity_delta: torch.Tensor,
        threshold: float = 0.4
    ) -> List[EventToken]:
        """
        Extract change events from mask and integrity scores.
        
        Args:
            change_mask: Change mask [B, 1, H, W]
            integrity_delta: Integrity change scores [B, 1]
            threshold: Event detection threshold
            
        Returns:
            List of EventToken objects
        """
        B = change_mask.shape[0]
        events = []
        
        for b in range(B):
            mask = change_mask[b, 0]  # [H, W]
            delta = float(integrity_delta[b, 0].item())
            
            # Compute overall change score
            change_score = float(mask.mean().item())
            
            # Create event if above threshold
            if change_score >= threshold or abs(delta) >= threshold:
                event = EventToken(
                    type="rgb_change",
                    score=max(change_score, abs(delta)),
                    t_ms=0,  # Frame-based, no specific timing
                    meta={
                        'change_score': change_score,
                        'integrity_delta': delta,
                        'mask_max': float(mask.max().item()),
                        'batch_idx': b
                    }
                )
                events.append(event)
        
        return events
    
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
            'change_mask': (B, 1, H, W),
            'integrity_delta': (B, 1),
            'zi': (B, self.out_embed_dim),
            'events': "List[EventToken]"
        }