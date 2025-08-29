"""
TRIDENT-R3: RadarFormer (Transformer for radar tokens)

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..common.types import BranchModule, EventToken, FeatureVec


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class RadarTokenProcessor(nn.Module):
    """Process radar tokens for transformer input."""
    
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Token type embeddings for different radar token types
        self.token_type_embedding = nn.Embedding(4, d_model)  # range, doppler, power, phase
        
    def forward(self, tokens: torch.Tensor, token_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process radar tokens.
        
        Args:
            tokens: Raw radar tokens (B, T, D)
            token_types: Optional token type indices (B, T)
            
        Returns:
            Processed tokens (B, T, d_model)
        """
        # Project to model dimension
        tokens = self.input_projection(tokens)
        
        # Add token type embeddings if provided
        if token_types is not None:
            type_emb = self.token_type_embedding(token_types)
            tokens = tokens + type_emb
        
        return self.layer_norm(tokens)


class RadarAttentionHead(nn.Module):
    """Specialized attention head for radar pattern analysis."""
    
    def __init__(self, d_model: int, num_patterns: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.num_patterns = num_patterns
        
        # Pattern query vectors
        self.pattern_queries = nn.Parameter(torch.randn(num_patterns, d_model) * 0.02)
        
        # Attention computation
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        
        # Pattern classification
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_patterns),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply radar-specific attention.
        
        Args:
            sequence: Input sequence (B, T, d_model)
            
        Returns:
            tuple: (attended_sequence, pattern_scores, attention_weights)
        """
        B, T, D = sequence.shape
        
        # Expand pattern queries for batch
        queries = self.pattern_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_patterns, d_model)
        
        # Apply attention with pattern queries
        attended_out, attention_weights = self.attention(
            queries, sequence, sequence
        )  # (B, num_patterns, d_model), (B, num_patterns, T)
        
        # Classify patterns
        pattern_scores = self.pattern_classifier(attended_out.mean(dim=1))  # (B, num_patterns)
        
        # Combine attended patterns back to sequence
        # Use attention weights to create attended sequence
        pattern_influence = torch.bmm(attention_weights, sequence)  # (B, num_patterns, d_model)
        attended_sequence = pattern_influence.mean(dim=1, keepdim=True).expand(-1, T, -1)
        
        return attended_sequence, pattern_scores, attention_weights


class RadarFormer(BranchModule):
    """
    RadarFormer: Transformer encoder for radar token sequences.
    
    Processes tokenized radar features using transformer architecture
    with radar-specific attention mechanisms.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 6,
        out_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__(out_dim)
        
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.max_seq_len = max_seq_len
        
        # Input dimension will be inferred
        self.input_dim = None
        self.token_processor = None
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # Radar-specific attention
        self.radar_attention = RadarAttentionHead(d_model)
        
        # Output heads
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.feature_head = nn.Sequential(
            nn.Linear(d_model // 2, out_dim),
        )
        
        # Event detection heads
        self.event_detectors = nn.ModuleDict({
            'target_classification': nn.Sequential(
                nn.Linear(out_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 4),  # vehicle, aircraft, drone, unknown
                nn.Softmax(dim=1),
            ),
            'multipath_detected': nn.Sequential(
                nn.Linear(out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ),
            'clutter_level': nn.Sequential(
                nn.Linear(out_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
        })
        
        self._initialized = False
    
    def _initialize_token_processor(self, input_dim: int) -> None:
        """Initialize token processor based on input dimension."""
        self.input_dim = input_dim
        self.token_processor = RadarTokenProcessor(input_dim, self.d_model)
        self._initialized = True
    
    def forward(self, rd_tokens: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for radar transformer.
        
        Args:
            rd_tokens: Radar tokens of shape (B, T, D) where:
                      T = sequence length, D = token dimension
                      
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        if not self._initialized:
            self._initialize_token_processor(rd_tokens.shape[-1])
        
        B, T, D = rd_tokens.shape
        
        # Limit sequence length if needed
        if T > self.max_seq_len:
            indices = torch.linspace(0, T-1, self.max_seq_len, dtype=torch.long, device=rd_tokens.device)
            rd_tokens = rd_tokens[:, indices]
            T = self.max_seq_len
        
        # Process tokens
        tokens = self.token_processor(rd_tokens)  # (B, T, d_model)
        
        # Add positional encoding
        tokens = tokens.transpose(0, 1)  # (T, B, d_model)
        tokens = self.pos_encoding(tokens)
        tokens = tokens.transpose(0, 1)  # (B, T, d_model)
        
        # Apply transformer
        transformer_out = self.transformer(tokens)  # (B, T, d_model)
        
        # Apply radar-specific attention
        attended_seq, pattern_scores, attention_weights = self.radar_attention(transformer_out)
        
        # Global pooling (use both mean and max)
        mean_pooled = transformer_out.mean(dim=1)  # (B, d_model)
        max_pooled, _ = transformer_out.max(dim=1)  # (B, d_model)
        combined_pooled = (mean_pooled + max_pooled) / 2
        
        # Project to output features
        pooled_features = self.global_pool(combined_pooled)  # (B, d_model//2)
        global_features = self.feature_head(pooled_features)  # (B, out_dim)
        
        # Event detection
        events_output = {}
        for event_type, detector in self.event_detectors.items():
            events_output[event_type] = detector(global_features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=global_features)
        
        # Create EventTokens
        events = []
        for b in range(B):
            # Target classification event
            target_probs = events_output['target_classification'][b]
            target_idx = target_probs.argmax().item()
            target_conf = target_probs[target_idx].item()
            
            if target_conf > 0.6:
                target_types = ["vehicle", "aircraft", "drone", "unknown"]
                target_type = target_types[target_idx]
                
                events.append(EventToken(
                    type=f"target_{target_type}",
                    value=target_conf,
                    t_start=0.0,
                    t_end=T / 100.0,  # Assume 100Hz token rate
                    quality=target_conf,
                    meta={
                        "target_type": target_type,
                        "classification_confidence": target_conf,
                        "pattern_scores": pattern_scores[b].tolist(),
                        "sequence_length": T,
                    }
                ))
            
            # Multipath detection event
            multipath_prob = events_output['multipath_detected'][b, 0].item()
            if multipath_prob > 0.7:
                events.append(EventToken(
                    type="multipath_detected",
                    value=multipath_prob,
                    t_start=0.0,
                    t_end=T / 100.0,
                    quality=multipath_prob,
                    meta={
                        "multipath_probability": multipath_prob,
                        "signal_complexity": pattern_scores[b].max().item(),
                    }
                ))
            
            # Clutter level event
            clutter_level = events_output['clutter_level'][b, 0].item()
            clutter_normalized = torch.sigmoid(torch.tensor(clutter_level)).item()
            
            if clutter_normalized > 0.8:
                events.append(EventToken(
                    type="high_clutter",
                    value=clutter_normalized,
                    t_start=0.0,
                    t_end=T / 100.0,
                    quality=clutter_normalized,
                    meta={
                        "clutter_level": clutter_level,
                        "normalized_clutter": clutter_normalized,
                        "environment": "dense" if clutter_normalized > 0.9 else "moderate",
                    }
                ))
        
        # Store intermediate outputs
        self._last_transformer_out = transformer_out
        self._last_attention_weights = attention_weights
        self._last_pattern_scores = pattern_scores
        self._last_events_raw = events_output
        
        return feature_vec, events
    
    def get_transformer_features(self) -> torch.Tensor:
        """Get transformer output features."""
        if hasattr(self, '_last_transformer_out'):
            return self._last_transformer_out
        else:
            raise RuntimeError("No transformer features available. Run forward() first.")
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get radar attention weights."""
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        else:
            raise RuntimeError("No attention weights available. Run forward() first.")
    
    def get_pattern_scores(self) -> torch.Tensor:
        """Get radar pattern classification scores."""
        if hasattr(self, '_last_pattern_scores'):
            return self._last_pattern_scores
        else:
            raise RuntimeError("No pattern scores available. Run forward() first.")
    
    def get_event_scores(self) -> dict:
        """Get raw event detection scores."""
        if hasattr(self, '_last_events_raw'):
            return self._last_events_raw
        else:
            raise RuntimeError("No event scores available. Run forward() first.")


def create_radar_former(config: dict) -> RadarFormer:
    """Factory function to create RadarFormer from config."""
    return RadarFormer(
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 8),
        nlayers=config.get("nlayers", 6),
        out_dim=config.get("out_dim", 256),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 1024),
    )