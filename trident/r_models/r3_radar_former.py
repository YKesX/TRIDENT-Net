"""
RadarFormer: Transformer encoder for tokenized radar features.

Processes tokenized radar features using transformer architecture
for long-range dependency modeling and complex radar signature analysis.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..common.types import BranchModule, EventToken, FeatureVec


class RadarFormer(BranchModule):
    """
    Radar transformer for complex radar signature processing.
    
    Uses transformer encoder architecture to process tokenized radar features
    and capture long-range dependencies in radar signatures.
    """
    
    def __init__(
        self,
        token_dim: int = 64,
        out_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_sequence_length: int = 512,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize RadarFormer.
        
        Args:
            token_dim: Input token dimension
            out_dim: Output feature dimension
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_sequence_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Token embedding and projection
        self.token_embedding = nn.Linear(token_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            hidden_dim, max_sequence_length, dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Multi-head attention for global feature aggregation
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        # Event detection heads
        self.target_classifier = RadarTargetClassifier(hidden_dim)
        self.anomaly_detector = RadarAnomalyDetector(hidden_dim)
        self.signature_analyzer = RadarSignatureAnalyzer(hidden_dim)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, rd_tokens: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process tokenized radar features.
        
        Args:
            rd_tokens: Radar tokens (B, T, token_dim)
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size, seq_len, _ = rd_tokens.shape
        
        # Token embedding
        embedded_tokens = self.token_embedding(rd_tokens)
        
        # Add special tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        
        # Concatenate: [CLS] + tokens + [SEP]
        sequence = torch.cat([cls_tokens, embedded_tokens, sep_tokens], dim=1)
        
        # Add positional encoding
        sequence = self.positional_encoding(sequence)
        
        # Transformer processing
        transformer_output = self.transformer(sequence)
        
        # Extract CLS token for global representation
        cls_output = transformer_output[:, 0]  # (B, hidden_dim)
        
        # Apply global attention for better aggregation
        attended_output, attention_weights = self.global_attention(
            cls_output.unsqueeze(1),
            transformer_output,
            transformer_output
        )
        global_features = attended_output.squeeze(1)  # (B, hidden_dim)
        
        # Feature projection
        projected_features = self.feature_proj(global_features)
        
        # Event detection using full transformer output
        target_events = self.target_classifier(transformer_output, attention_weights)
        anomaly_events = self.anomaly_detector(transformer_output, attention_weights)
        signature_events = self.signature_analyzer(transformer_output, attention_weights)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Combine all events
        events = []
        for b in range(batch_size):
            # Add target events
            for event in target_events:
                if event["batch_idx"] == b:
                    events.append(EventToken(
                        type=event["type"],
                        value=event["value"],
                        t_start=event["t_start"],
                        t_end=event["t_end"],
                        quality=event["quality"],
                        meta=event["meta"]
                    ))
            
            # Add anomaly events
            for event in anomaly_events:
                if event["batch_idx"] == b:
                    events.append(EventToken(
                        type=event["type"],
                        value=event["value"],
                        t_start=event["t_start"],
                        t_end=event["t_end"],
                        quality=event["quality"],
                        meta=event["meta"]
                    ))
            
            # Add signature events
            for event in signature_events:
                if event["batch_idx"] == b:
                    events.append(EventToken(
                        type=event["type"],
                        value=event["value"],
                        t_start=event["t_start"],
                        t_end=event["t_end"],
                        quality=event["quality"],
                        meta=event["meta"]
                    ))
        
        return feature_vec, events
    
    def get_attention_maps(self, rd_tokens: torch.Tensor) -> torch.Tensor:
        """
        Get transformer attention maps for visualization.
        
        Args:
            rd_tokens: Radar tokens (B, T, token_dim)
            
        Returns:
            Attention maps from all layers (num_layers, B, num_heads, T+2, T+2)
        """
        batch_size, seq_len, _ = rd_tokens.shape
        
        # Prepare input sequence
        embedded_tokens = self.token_embedding(rd_tokens)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, embedded_tokens, sep_tokens], dim=1)
        sequence = self.positional_encoding(sequence)
        
        # Extract attention maps from transformer layers
        attention_maps = []
        x = sequence
        
        for layer in self.transformer.layers:
            # Get attention weights from multi-head attention
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_maps.append(attn_weights)
            
            # Continue through the layer
            x = layer(x)
        
        return torch.stack(attention_maps, dim=0)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(
        self,
        d_model: int,
        max_length: int = 5000,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (B, T, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class RadarTargetClassifier(nn.Module):
    """Classify radar target types from transformer features."""
    
    def __init__(self, hidden_dim: int) -> None:
        """
        Initialize radar target classifier.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # 4 target types
            nn.Softmax(dim=-1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        transformer_output: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> List[dict]:
        """
        Classify radar targets.
        
        Args:
            transformer_output: Transformer output (B, T+2, hidden_dim)
            attention_weights: Attention weights (B, 1, T+2)
            
        Returns:
            List of target events
        """
        batch_size = transformer_output.shape[0]
        events = []
        
        # Use CLS token for classification
        cls_features = transformer_output[:, 0]  # (B, hidden_dim)
        
        target_probs = self.target_head(cls_features)  # (B, 4)
        confidence = self.confidence_head(cls_features)  # (B, 1)
        
        target_types = ["aircraft", "vehicle", "structure", "clutter"]
        
        for b in range(batch_size):
            max_prob_idx = torch.argmax(target_probs[b]).item()
            max_prob = target_probs[b, max_prob_idx].item()
            conf = confidence[b].item()
            
            if max_prob > 0.6 and conf > 0.7:
                events.append({
                    "type": f"target_{target_types[max_prob_idx]}",
                    "value": max_prob,
                    "t_start": 0.0,
                    "t_end": transformer_output.shape[1] - 2,  # Exclude special tokens
                    "quality": conf,
                    "batch_idx": b,
                    "meta": {
                        "source": "radar_target_classifier",
                        "target_type": target_types[max_prob_idx],
                        "classification_confidence": conf,
                        "type_probabilities": target_probs[b].tolist()
                    }
                })
        
        return events


class RadarAnomalyDetector(nn.Module):
    """Detect anomalies in radar signatures."""
    
    def __init__(self, hidden_dim: int) -> None:
        """
        Initialize radar anomaly detector.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.anomaly_score = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_localizer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        transformer_output: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> List[dict]:
        """
        Detect radar anomalies.
        
        Args:
            transformer_output: Transformer output (B, T+2, hidden_dim)
            attention_weights: Attention weights (B, 1, T+2)
            
        Returns:
            List of anomaly events
        """
        batch_size, seq_len, _ = transformer_output.shape
        events = []
        
        # Compute anomaly scores for each token
        token_features = transformer_output[:, 1:-1]  # Exclude special tokens
        anomaly_scores = self.anomaly_score(token_features)  # (B, T, 1)
        localization_scores = self.anomaly_localizer(token_features)  # (B, T, 1)
        
        for b in range(batch_size):
            token_anomalies = anomaly_scores[b].squeeze(-1)  # (T,)
            token_localizations = localization_scores[b].squeeze(-1)  # (T,)
            
            # Find anomalous regions
            anomaly_threshold = 0.7
            anomalous_tokens = token_anomalies > anomaly_threshold
            
            if anomalous_tokens.any():
                # Find continuous anomalous regions
                anomaly_indices = torch.where(anomalous_tokens)[0]
                
                # Group continuous indices
                regions = []
                current_region = [anomaly_indices[0].item()]
                
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] - anomaly_indices[i-1] == 1:
                        current_region.append(anomaly_indices[i].item())
                    else:
                        regions.append(current_region)
                        current_region = [anomaly_indices[i].item()]
                regions.append(current_region)
                
                # Create events for each region
                for region in regions:
                    start_time = region[0]
                    end_time = region[-1] + 1
                    region_scores = token_anomalies[region]
                    region_localizations = token_localizations[region]
                    
                    events.append({
                        "type": "radar_anomaly",
                        "value": region_scores.mean().item(),
                        "t_start": start_time,
                        "t_end": end_time,
                        "quality": region_localizations.mean().item(),
                        "batch_idx": b,
                        "meta": {
                            "source": "radar_anomaly_detector",
                            "anomaly_strength": region_scores.max().item(),
                            "affected_tokens": len(region),
                            "localization_confidence": region_localizations.mean().item()
                        }
                    })
        
        return events


class RadarSignatureAnalyzer(nn.Module):
    """Analyze complex radar signatures."""
    
    def __init__(self, hidden_dim: int) -> None:
        """
        Initialize radar signature analyzer.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.signature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        self.complexity_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.stability_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        transformer_output: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> List[dict]:
        """
        Analyze radar signatures.
        
        Args:
            transformer_output: Transformer output (B, T+2, hidden_dim)
            attention_weights: Attention weights (B, 1, T+2)
            
        Returns:
            List of signature events
        """
        batch_size = transformer_output.shape[0]
        events = []
        
        # Extract signature features
        cls_features = transformer_output[:, 0]  # (B, hidden_dim)
        signature_features = self.signature_extractor(cls_features)  # (B, 64)
        
        complexity = self.complexity_estimator(signature_features)  # (B, 1)
        stability = self.stability_estimator(signature_features)  # (B, 1)
        
        for b in range(batch_size):
            complex_score = complexity[b].item()
            stable_score = stability[b].item()
            
            # High complexity signature
            if complex_score > 0.8:
                events.append({
                    "type": "complex_signature",
                    "value": complex_score,
                    "t_start": 0.0,
                    "t_end": transformer_output.shape[1] - 2,
                    "quality": complex_score,
                    "batch_idx": b,
                    "meta": {
                        "source": "radar_signature_analyzer",
                        "complexity_score": complex_score,
                        "stability_score": stable_score
                    }
                })
            
            # Unstable signature
            if stable_score < 0.3:
                events.append({
                    "type": "unstable_signature",
                    "value": 1.0 - stable_score,
                    "t_start": 0.0,
                    "t_end": transformer_output.shape[1] - 2,
                    "quality": 1.0 - stable_score,
                    "batch_idx": b,
                    "meta": {
                        "source": "radar_signature_analyzer",
                        "instability_score": 1.0 - stable_score,
                        "complexity_score": complex_score
                    }
                })
        
        return events


# Default configuration
RADAR_FORMER_CONFIG = {
    "token_dim": 64,
    "out_dim": 256,
    "hidden_dim": 512,
    "num_layers": 6,
    "num_heads": 8,
    "max_sequence_length": 512,
    "dropout": 0.1
}