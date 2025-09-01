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
                 mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__(out_dim=512)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projections for each modality
        self.i_proj = nn.Linear(768, d_model)  # I-branch: 768 -> 512
        self.t_proj = nn.Linear(512, d_model)  # T-branch: 512 -> 512  
        self.r_proj = nn.Linear(384, d_model)  # R-branch: 384 -> 512
        
        # Modality embeddings
        self.modality_embeddings = nn.Parameter(torch.randn(3, d_model))
        
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
                events: Optional[List[EventToken]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, List]:
        """
        Forward pass through CrossAttnFusion.
        
        Args:
            zi: I-branch features (B, 768)
            zt: T-branch features (B, 512)
            zr: R-branch features (B, 384)
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
        
        # Add modality embeddings
        zi_proj = zi_proj + self.modality_embeddings[0]
        zt_proj = zt_proj + self.modality_embeddings[1]
        zr_proj = zr_proj + self.modality_embeddings[2]
        
        # Stack modalities as sequence (B, 3, d_model)
        modality_sequence = torch.stack([zi_proj, zt_proj, zr_proj], dim=1)
        
        # Apply transformer layers with attention tracking
        attention_maps = {}
        x = modality_sequence
        
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