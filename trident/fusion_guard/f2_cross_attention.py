"""
TRIDENT-F2: Cross-Attention Fusion (multimodal transformer)

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..common.types import FusionModule, FeatureVec, OutcomeEstimate, EventToken


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention between modalities."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-attention.
        
        Args:
            query: Query tensor (B, seq_len_q, d_model)
            key: Key tensor (B, seq_len_k, d_model)
            value: Value tensor (B, seq_len_v, d_model)
            mask: Optional attention mask
            
        Returns:
            tuple: (attended_output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attended)
        
        # Return average attention weights across heads for visualization
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class ModalityEmbedding(nn.Module):
    """Embed features with modality-specific information."""
    
    def __init__(self, feature_dim: int, d_model: int, modality_types: int = 3):
        super().__init__()
        
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.modality_embedding = nn.Embedding(modality_types, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, features: torch.Tensor, modality_id: int) -> torch.Tensor:
        """
        Embed features with modality information.
        
        Args:
            features: Input features (B, feature_dim)
            modality_id: Modality identifier (0=radar, 1=visible, 2=thermal)
            
        Returns:
            Embedded features (B, 1, d_model)
        """
        # Project features
        projected = self.feature_projection(features)  # (B, d_model)
        
        # Add modality embedding
        batch_size = features.size(0)
        modality_emb = self.modality_embedding(
            torch.full((batch_size,), modality_id, device=features.device)
        )
        
        # Combine and normalize
        embedded = self.layer_norm(projected + modality_emb)
        
        return embedded.unsqueeze(1)  # (B, 1, d_model)


class FusionTransformerBlock(nn.Module):
    """Single transformer block for multimodal fusion."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion transformer block.
        
        Args:
            x: Input sequence (B, seq_len, d_model)
            context: Context from other modalities (B, context_len, d_model)
            
        Returns:
            tuple: (output, cross_attention_weights)
        """
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention with context
        cross_out, cross_attn = self.cross_attention(x, context, context)
        x = self.norm2(x + self.dropout(cross_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x, cross_attn


class CrossAttentionFusion(FusionModule):
    """
    Cross-Attention Fusion using transformer architecture.
    
    Fuses multimodal features using cross-attention mechanisms
    to model inter-modal dependencies and interactions.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        nlayers: int = 4,
        dropout: float = 0.1,
        input_feature_dim: int = 256,
    ):
        super().__init__(out_dim=d_model)
        
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.input_feature_dim = input_feature_dim
        
        # Modality embeddings
        self.radar_embedding = ModalityEmbedding(input_feature_dim, d_model, 3)
        self.visible_embedding = ModalityEmbedding(input_feature_dim, d_model, 3)
        self.thermal_embedding = ModalityEmbedding(input_feature_dim, d_model, 3)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FusionTransformerBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(nlayers)
        ])
        
        # Event processing
        self.event_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output heads
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )
        
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        
        # For storing attention maps
        self.attention_maps = {}
    
    def _process_events(self, events: List[EventToken]) -> torch.Tensor:
        """
        Process event tokens into feature representation.
        
        Args:
            events: List of event tokens
            
        Returns:
            Event features (B, d_model)
        """
        if not events:
            # Return zero features if no events
            return torch.zeros(1, self.d_model)
        
        # Simple event encoding (can be made more sophisticated)
        event_features = []
        
        # Group events by type and compute statistics
        event_types = {}
        for event in events:
            if event.type not in event_types:
                event_types[event.type] = []
            
            # Convert value to float if needed
            if isinstance(event.value, torch.Tensor):
                value = event.value.item()
            else:
                value = float(event.value)
            
            event_types[event.type].append({
                'value': value,
                'quality': event.quality,
                'duration': event.t_end - event.t_start,
            })
        
        # Create feature vector from event statistics
        feature_vector = []
        
        # Number of event types
        feature_vector.append(len(event_types))
        
        # Total number of events
        feature_vector.append(len(events))
        
        # Average event quality
        avg_quality = sum(e.quality for e in events) / len(events)
        feature_vector.append(avg_quality)
        
        # Event type diversity (entropy)
        type_counts = [len(event_types[t]) for t in event_types]
        total_events = sum(type_counts)
        type_probs = [c / total_events for c in type_counts]
        entropy = -sum(p * math.log(p + 1e-8) for p in type_probs)
        feature_vector.append(entropy)
        
        # Pad to d_model dimensions
        while len(feature_vector) < self.d_model:
            feature_vector.append(0.0)
        
        return torch.tensor(feature_vector[:self.d_model], dtype=torch.float32)
    
    def forward(
        self,
        z_r: Optional[FeatureVec] = None,
        z_i: Optional[FeatureVec] = None,
        z_t: Optional[FeatureVec] = None,
        events: Optional[List[EventToken]] = None,
    ) -> OutcomeEstimate:
        """
        Forward pass for cross-attention fusion.
        
        Args:
            z_r: Radar features
            z_i: Visible/EO features
            z_t: Thermal/IR features
            events: Event tokens
            
        Returns:
            OutcomeEstimate with fused predictions and attention maps
        """
        # Collect available modalities
        modality_features = []
        modality_names = []
        attention_maps = {}
        
        if z_r is not None:
            radar_emb = self.radar_embedding(z_r.z, modality_id=0)
            modality_features.append(radar_emb)
            modality_names.append("radar")
        
        if z_i is not None:
            visible_emb = self.visible_embedding(z_i.z, modality_id=1)
            modality_features.append(visible_emb)
            modality_names.append("visible")
        
        if z_t is not None:
            thermal_emb = self.thermal_embedding(z_t.z, modality_id=2)
            modality_features.append(thermal_emb)
            modality_names.append("thermal")
        
        if not modality_features:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all modalities
        fused_sequence = torch.cat(modality_features, dim=1)  # (B, num_modalities, d_model)
        batch_size = fused_sequence.size(0)
        
        # Apply transformer blocks with cross-attention
        for i, block in enumerate(self.transformer_blocks):
            # Use the sequence as both input and context for cross-modal attention
            fused_sequence, attn_weights = block(fused_sequence, fused_sequence)
            
            # Store attention weights for visualization
            attention_maps[f"layer_{i}"] = attn_weights.detach()
        
        # Global pooling across modalities
        pooled_features = fused_sequence.mean(dim=1)  # (B, d_model)
        
        # Process events if provided
        if events:
            event_features = self._process_events(events)
            if event_features.dim() == 1:
                event_features = event_features.unsqueeze(0).expand(batch_size, -1)
            
            # Combine with pooled features
            event_processed = self.event_processor(event_features)
            pooled_features = pooled_features + event_processed
        
        # Generate final fused features
        z_fused = self.fusion_head(pooled_features)
        
        # Predict outcome
        p_outcome = self.outcome_head(z_fused).squeeze(-1)
        binary_outcome = (p_outcome > 0.5).long()
        
        # Create explanation with attention maps
        explanation = {
            "fusion_type": "cross_attention",
            "modalities_used": modality_names,
            "attention_maps": {
                name: attn.cpu().numpy().tolist() 
                for name, attn in attention_maps.items()
            },
            "fused_feature_norm": torch.norm(z_fused, dim=-1).tolist(),
            "modality_contributions": self._compute_modality_contributions(
                modality_features, attention_maps
            ),
        }
        
        # Store attention maps for later access
        self.attention_maps = attention_maps
        
        return OutcomeEstimate(
            p_outcome=p_outcome,
            binary_outcome=binary_outcome,
            explanation=explanation,
        )
    
    def _compute_modality_contributions(
        self, 
        modality_features: List[torch.Tensor],
        attention_maps: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute relative contributions of each modality."""
        if not attention_maps:
            # Equal contribution if no attention maps
            return {f"modality_{i}": 1.0/len(modality_features) 
                   for i in range(len(modality_features))}
        
        # Use last layer attention for contribution analysis
        last_attn = attention_maps[list(attention_maps.keys())[-1]]
        
        # Average attention weights across batch and heads
        avg_attn = last_attn.mean(dim=0)  # Average across batch
        
        # Contribution is the sum of attention weights for each modality
        contributions = {}
        num_modalities = len(modality_features)
        
        for i in range(num_modalities):
            contrib = avg_attn[:, i].sum().item()
            contributions[f"modality_{i}"] = contrib
        
        # Normalize contributions
        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            contributions = {k: v/total_contrib for k, v in contributions.items()}
        
        return contributions
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get stored attention maps from last forward pass."""
        return self.attention_maps


def create_cross_attention_fusion(config: dict) -> CrossAttentionFusion:
    """Factory function to create CrossAttentionFusion from config."""
    return CrossAttentionFusion(
        d_model=config.get("d_model", 384),
        nhead=config.get("nhead", 8),
        nlayers=config.get("nlayers", 4),
        dropout=config.get("dropout", 0.1),
        input_feature_dim=config.get("input_feature_dim", 256),
    )