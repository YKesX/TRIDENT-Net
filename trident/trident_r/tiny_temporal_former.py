"""
TRIDENT-R3: TinyTempoFormer - 2-layer temporal transformer over 3 kinematic tokens

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..common.types import BranchModule, EventToken


class TinyTempoFormer(BranchModule):
    """
    2-layer temporal transformer over 3 tokenized steps.
    
    Applies a compact transformer to capture temporal patterns in
    kinematic sequences, projecting per-step features to tokens.
    
    Input: k_tokens (B, 3, 32) - 3 tokens of 32-D each  
    Outputs:
        - zr3 (B, 192) - transformer output embedding
        - events (list) - temporal pattern events
    """
    
    def __init__(self, d_model: int = 192, n_heads: int = 4, n_layers: int = 2,
                 token_dim: int = 32, dropout: float = 0.1):
        super().__init__(out_dim=192)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.token_dim = token_dim
        
        # Token projection: project 32-D tokens to d_model
        self.token_projection = nn.Linear(token_dim, d_model)
        
        # Positional encoding for 3 time steps
        self.pos_encoding = PositionalEncoding(d_model, max_len=3)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # Flatten temporal dimension
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Attention analysis for event detection
        self.attention_analyzer = AttentionAnalyzer(d_model, n_heads)
        
    def forward(self, k_tokens: torch.Tensor) -> Tuple[torch.Tensor, List[EventToken]]:
        """
        Forward pass through TinyTempoFormer.
        
        Args:
            k_tokens: Kinematic tokens (B, 3, 32)
                     3 time steps × 32-D token features each
                     
        Returns:
            tuple: (zr3, events)
                - zr3: (B, 192) transformer output embedding
                - events: List of temporal pattern events
        """
        B, T, D = k_tokens.shape
        assert T == 3 and D == 32, f"Expected k_tokens shape (B, 3, 32), got {k_tokens.shape}"
        
        # Project tokens to model dimension
        tokens = self.token_projection(k_tokens)  # (B, 3, d_model)
        
        # Add positional encoding
        tokens = self.pos_encoding(tokens)  # (B, 3, d_model)
        
        # Apply transformer with attention tracking
        transformer_output, attention_weights = self._forward_with_attention(tokens)  # (B, 3, d_model)
        
        # Flatten temporal dimension and project to output
        flattened = transformer_output.view(B, -1)  # (B, 3 * d_model)
        zr3 = self.output_projection(flattened)  # (B, 192)
        
        # Analyze attention patterns for events
        events = self._extract_temporal_events(k_tokens, transformer_output, attention_weights)
        
        return zr3, events
    
    def _forward_with_attention(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass while capturing attention weights."""
        B, T, D = tokens.shape
        
        # Store attention weights from each layer
        attention_weights_list = []
        
        x = tokens
        for layer in self.transformer.layers:
            # Manual forward through transformer layer to capture attention
            # Pre-norm
            x_norm = layer.norm1(x)
            
            # Self-attention
            attn_output, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm, 
                need_weights=True, average_attn_weights=True
            )
            attention_weights_list.append(attn_weights)
            
            # Residual connection
            x = x + layer.dropout1(attn_output)
            
            # Feed-forward
            x_norm2 = layer.norm2(x)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm2))))
            x = x + layer.dropout2(ff_output)
        
        # Average attention weights across layers and heads
        avg_attention = torch.stack(attention_weights_list, dim=0).mean(dim=0)  # (B, T, T)
        
        return x, avg_attention
    
    def _extract_temporal_events(self, k_tokens: torch.Tensor, transformer_output: torch.Tensor,
                                attention_weights: torch.Tensor) -> List[EventToken]:
        """Extract temporal pattern events from transformer analysis."""
        events = []
        B, T, D = k_tokens.shape
        
        for b in range(B):
            batch_tokens = k_tokens[b]  # (3, 32)
            batch_output = transformer_output[b]  # (3, d_model)
            batch_attention = attention_weights[b]  # (3, 3)
            
            # Event 1: High self-attention (temporal persistence)
            self_attention_scores = torch.diag(batch_attention)  # (3,)
            max_self_attention = self_attention_scores.max()
            
            if max_self_attention > 0.7:
                event = EventToken(
                    type="temporal_persistence",
                    score=max_self_attention.item(),
                    t_ms=self_attention_scores.argmax().item() * 100,  # Convert to ms
                    meta={
                        'self_attention_score': max_self_attention.item(),
                        'persistence_time': self_attention_scores.argmax().item(),
                        'location': (0, 0),
                        'source': "tinytempformer",
                        'batch_idx': b
                    }
                )
                events.append(event)
            
            # Event 2: High cross-temporal attention (temporal dependency)
            # Sum of off-diagonal attention weights
            off_diag_mask = ~torch.eye(3, dtype=torch.bool)
            cross_attention = batch_attention[off_diag_mask].sum()
            
            if cross_attention > 1.0:  # High cross-temporal dependency
                # Find strongest cross-temporal connection
                batch_attention_masked = batch_attention.clone()
                batch_attention_masked.fill_diagonal_(0)
                max_cross_idx = torch.unravel_index(batch_attention_masked.argmax(), batch_attention_masked.shape)
                
                event = EventToken(
                    type="temporal_dependency",
                    score=min(1.0, cross_attention.item() / 2.0),
                    t_ms=max_cross_idx[0].item() * 100,  # Source timestamp, convert to ms
                    meta={
                        'cross_attention_sum': cross_attention.item(),
                        'strongest_connection': (max_cross_idx[0].item(), max_cross_idx[1].item()),
                        'connection_strength': batch_attention_masked.max().item(),
                        'location': (0, 0),
                        'source': "tinytempformer",
                        'batch_idx': b
                    }
                )
                events.append(event)
            
            # Event 3: Temporal gradient analysis (rapid changes)
            # Compute feature differences between consecutive time steps
            token_diff_01 = torch.norm(batch_tokens[1] - batch_tokens[0])
            token_diff_12 = torch.norm(batch_tokens[2] - batch_tokens[1])
            max_change = max(token_diff_01, token_diff_12)
            
            if max_change > 5.0:  # Significant temporal change threshold
                change_time = 1 if token_diff_01 > token_diff_12 else 2
                
                event = EventToken(
                    type="rapid_temporal_change",
                    score=min(1.0, max_change.item() / 10.0),
                    t_ms=change_time * 100,  # Convert to ms
                    meta={
                        'max_change_magnitude': max_change.item(),
                        'change_01': token_diff_01.item(),
                        'change_12': token_diff_12.item(),
                        'change_time': change_time,
                        'location': (0, 0),
                        'source': "tinytempformer",
                        'batch_idx': b
                    }
                )
                events.append(event)
            
            # Event 4: Output embedding analysis (temporal complexity)
            output_variance = torch.var(batch_output, dim=0).mean()  # Variance across time
            output_magnitude = torch.norm(batch_output, dim=1).mean()  # Average magnitude
            
            if output_variance > 2.0 or output_magnitude > 8.0:
                event = EventToken(
                    type="high_temporal_complexity",
                    score=min(1.0, (output_variance.item() + output_magnitude.item()) / 20.0),
                    t_ms=100,  # Middle of sequence, convert to ms
                    meta={
                        'output_variance': output_variance.item(),
                        'output_magnitude': output_magnitude.item(),
                        'temporal_complexity_score': (output_variance + output_magnitude).item(),
                        'location': (0, 0),
                        'source': "tinytempformer",
                        'batch_idx': b
                    }
                )
                events.append(event)
        
        return events


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 3):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tokens."""
        return x + self.pe[:, :x.size(1)]


class AttentionAnalyzer(nn.Module):
    """Analyze attention patterns for interpretability."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze attention patterns for different temporal behaviors."""
        B, T, T = attention_weights.shape
        
        patterns = {}
        
        # Self-attention strength (diagonal elements)
        patterns['self_attention'] = torch.diag_embed(torch.diagonal(attention_weights, dim1=-2, dim2=-1))
        
        # Forward attention (upper triangular)
        forward_mask = torch.triu(torch.ones(T, T), diagonal=1)
        patterns['forward_attention'] = attention_weights * forward_mask
        
        # Backward attention (lower triangular)  
        backward_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        patterns['backward_attention'] = attention_weights * backward_mask
        
        # Attention entropy (uniformity measure)
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        patterns['attention_entropy'] = attention_entropy
        
        return patterns