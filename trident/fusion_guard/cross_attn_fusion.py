"""
CrossAttnFusion - Cross-modal transformer fusion with multitask outputs

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..common.types import FusionModule, EventToken


class ClassEmbedding(nn.Module):
    """
    Class embedding layer for injecting class information into fusion.
    
    Creates embeddings for class IDs with fallback support for unknown classes.
    """
    
    def __init__(self, num_classes: int, embed_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Main embedding table for known classes
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
        # Fallback embedding for unknown/out-of-vocab classes  
        self.unknown_embedding = nn.Parameter(torch.randn(embed_dim))
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.1)
        nn.init.normal_(self.unknown_embedding, std=0.1)
        
    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through class embedding.
        
        Args:
            class_ids: Class IDs (B,) - integer tensor
            
        Returns:
            Class embeddings (B, embed_dim)
        """
        B = class_ids.shape[0]
        device = class_ids.device
        
        # Handle out-of-vocabulary class IDs
        valid_mask = (class_ids >= 0) & (class_ids < self.num_classes)
        
        # Initialize output with unknown embeddings
        output = self.unknown_embedding.unsqueeze(0).expand(B, -1).clone()
        
        if valid_mask.any():
            # Use learned embeddings for valid class IDs
            valid_ids = class_ids[valid_mask]
            valid_embeddings = self.embedding(valid_ids)
            output[valid_mask] = valid_embeddings
            
        return output


class CrossAttnFusion(FusionModule):
    """
    Cross-modal transformer fusion module with multitask outputs.
    
    Fuses features from I/T/R branches using cross-attention and produces
    both hit and kill predictions with attention maps and top events.
    
    Input:
        - zi (B, 768) - concat(i1.zi=256, i2.zi=256, i3.zi=256)
        - zt (B, 512) - concat(t1.zt=256, t2.zt=256)  
        - zr (B, 384) - concat(r2.zr2=192, r3.zr3=192)
        - events (list) - events from all modalities
    Outputs:
        - z_fused (B, 512) - fused features
        - p_hit (B, 1) - hit probability
        - p_kill (B, 1) - kill probability
        - attn_maps (dict) - attention maps
        - top_events (list) - top contributing events
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 3,
                 mlp_hidden: int = 256, dropout: float = 0.1, 
                 dims: dict = None, num_classes: int = 100):
        super().__init__(out_dim=512)
        
        # Default dimensions if not provided
        if dims is None:
            dims = {'zi': 768, 'zt': 512, 'zr': 384, 'e_cls': 32}
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dims = dims
        
        # Input projections for each modality
        self.i_proj = nn.Linear(dims['zi'], d_model)  # I-branch: 768 -> 512
        self.t_proj = nn.Linear(dims['zt'], d_model)  # T-branch: 512 -> 512  
        self.r_proj = nn.Linear(dims['zr'], d_model)  # R-branch: 384 -> 512
        
        # Class embedding layer
        self.class_embedding = ClassEmbedding(num_classes, dims['e_cls'])
        self.cls_proj = nn.Linear(dims['e_cls'], d_model)  # Class: 32 -> 512
        
        # Modality embeddings (now 4 modalities: I, T, R, Class)
        self.modality_embeddings = nn.Parameter(torch.randn(4, d_model))
        
        # Cross-attention transformer layers
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Event processing
        self.event_processor = EventProcessor(d_model, dropout)
        
        # Output heads
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multitask prediction heads
        self.hit_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.kill_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, zi: torch.Tensor, zt: torch.Tensor, zr: torch.Tensor,
                class_ids: Optional[torch.Tensor] = None,
                events: Optional[List[EventToken]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, List]:
        """
        Forward pass through CrossAttnFusion.
        
        Args:
            zi: I-branch features (B, 768)
            zt: T-branch features (B, 512)
            zr: R-branch features (B, 384)
            class_ids: Class IDs (B,) - optional
            events: List of events from all modalities
            
        Returns:
            tuple: (z_fused, p_hit, p_kill, attn_maps, top_events)
        """
        B = zi.shape[0]
        device = zi.device
        
        # Project modality features to common dimension
        zi_proj = self.i_proj(zi)  # (B, d_model)
        zt_proj = self.t_proj(zt)  # (B, d_model)
        zr_proj = self.r_proj(zr)  # (B, d_model)
        
        # Handle class embeddings
        if class_ids is not None:
            class_emb = self.class_embedding(class_ids)  # (B, 32)
            zcls_proj = self.cls_proj(class_emb)  # (B, d_model)
        else:
            # Use zero embeddings if no class IDs provided
            zcls_proj = torch.zeros(B, self.d_model, device=device)
        
        # Stack modality features (I, T, R, Class)
        features = torch.stack([zi_proj, zt_proj, zr_proj, zcls_proj], dim=1)  # (B, 4, d_model)
        
        # Add modality embeddings to each feature
        features = features + self.modality_embeddings.unsqueeze(0)  # (B, 4, d_model)
        
        # Apply transformer layers with attention tracking
        attention_maps = {}
        x = features  # (B, 4, d_model)
        
        for i, layer in enumerate(self.transformer_layers):
            x, layer_attn = layer(x)
            attention_maps[f'layer_{i}'] = layer_attn
        
        # Process events if provided
        event_features = self.event_processor(events, B, device) if events else None
        
        # Global fusion with event integration
        if event_features is not None:
            # Add event features to sequence
            x = torch.cat([x, event_features.unsqueeze(1)], dim=1)  # (B, 4, d_model)
            
            # Final cross-attention with events
            final_layer = CrossModalTransformerLayer(self.d_model, self.n_heads, 
                                                   self.d_model * 4, 0.1)
            x, final_attn = final_layer(x)
            attention_maps['final_with_events'] = final_attn
        
        # Global pooling for fusion features
        z_fused = self.fusion_head(x.mean(dim=1))  # (B, d_model)
        
        # Multitask predictions
        p_hit = self.hit_head(z_fused)  # (B, 1)
        p_kill = self.kill_head(z_fused)  # (B, 1)
        
        # Extract top contributing events
        top_events = self._extract_top_events(events, attention_maps) if events else []
        
        return z_fused, p_hit, p_kill, attention_maps, top_events
    
    def _extract_top_events(self, events: List[EventToken], 
                           attention_maps: Dict) -> List[EventToken]:
        """Extract top contributing events based on attention."""
        if not events or not attention_maps:
            return []
        
        # Simple heuristic: return events with highest confidence
        sorted_events = sorted(events, key=lambda e: e.confidence, reverse=True)
        return sorted_events[:5]  # Top 5 events
    
    def get_calibration_features(self, zi: torch.Tensor, zt: torch.Tensor, zr: torch.Tensor,
                                class_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get concatenated features for calibration (zi + zt + zr + e_cls).
        
        Args:
            zi: I-branch features (B, 768)
            zt: T-branch features (B, 512)
            zr: R-branch features (B, 384)
            class_ids: Class IDs (B,) - optional
            
        Returns:
            Concatenated features (B, 1696)
        """
        # Get class embeddings
        if class_ids is not None:
            class_emb = self.class_embedding(class_ids)  # (B, 32)
        else:
            B = zi.shape[0]
            device = zi.device
            class_emb = torch.zeros(B, self.dims['e_cls'], device=device)
        
        # Concatenate all features
        calibration_features = torch.cat([zi, zt, zr, class_emb], dim=-1)  # (B, 1696)
        
        return calibration_features


class CrossModalTransformerLayer(nn.Module):
    """Single cross-modal transformer layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention tracking."""
        # Self-attention
        attn_output, attn_weights = self.self_attn(x, x, x, need_weights=True)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class EventProcessor(nn.Module):
    """Process events into feature representations."""
    
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        
        # Event type embedding
        self.event_types = [
            "debris_detection", "flash_detection", "structural_damage", 
            "thermal_signature", "close_approach", "high_acceleration"
        ]
        self.type_embedding = nn.Embedding(len(self.event_types), d_model // 4)
        
        # Event feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(d_model // 4 + 3, d_model // 2),  # type + [confidence, x, y]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, events: List[EventToken], batch_size: int, device: torch.device) -> torch.Tensor:
        """Process events into features."""
        if not events:
            return torch.zeros(batch_size, self.d_model, device=device)
        
        # Group events by batch
        batch_events = [[] for _ in range(batch_size)]
        for event in events:
            batch_idx = getattr(event.metadata, 'batch_idx', 0) if event.metadata else 0
            if batch_idx < batch_size:
                batch_events[batch_idx].append(event)
        
        # Process each batch's events
        batch_features = []
        for b_events in batch_events:
            if not b_events:
                batch_features.append(torch.zeros(self.d_model, device=device))
            else:
                # Take most confident event for simplicity
                top_event = max(b_events, key=lambda e: e.confidence)
                
                # Get type embedding
                type_idx = 0  # Default
                if top_event.event_type in self.event_types:
                    type_idx = self.event_types.index(top_event.event_type)
                
                type_emb = self.type_embedding(torch.tensor(type_idx, device=device))
                
                # Create feature vector
                location = top_event.location if top_event.location else (0, 0)
                features = torch.cat([
                    type_emb,
                    torch.tensor([top_event.confidence, location[0] / 640.0, location[1] / 480.0], 
                               device=device)
                ])
                
                batch_features.append(self.feature_processor(features))
        
        return torch.stack(batch_features, dim=0)  # (B, d_model)