"""
TRIDENT-R2: Pulse LSTM (BiLSTM for pulse features)

Author: Yağızhan Keskin
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken, FeatureVec


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Apply attention pooling to sequence.
        
        Args:
            sequence: Input sequence (B, T, H)
            lengths: Actual sequence lengths for masking
            
        Returns:
            Pooled features (B, H)
        """
        B, T, H = sequence.shape
        
        # Compute attention weights
        attn_weights = self.attention(sequence)  # (B, T, 1)
        
        # Apply length mask if provided
        if lengths is not None:
            mask = torch.arange(T, device=sequence.device).expand(B, T) < lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T, 1)
        
        # Apply attention
        pooled = (sequence * attn_weights).sum(dim=1)  # (B, H)
        
        return pooled


class PulseCharacteristics(nn.Module):
    """Analyze pulse characteristics from LSTM features."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Pulse repetition frequency analysis
        self.prf_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Pulse width analysis
        self.pw_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        # Amplitude modulation detection
        self.am_detector = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze pulse characteristics.
        
        Args:
            features: LSTM features (B, H)
            
        Returns:
            tuple: (prf_estimate, pulse_width, am_probability)
        """
        prf = self.prf_analyzer(features)
        pw = self.pw_analyzer(features)
        am_prob = self.am_detector(features)
        
        return prf, pw, am_prob


class PulseLSTM(BranchModule):
    """
    Pulse LSTM for processing radar pulse feature sequences.
    
    Uses bidirectional LSTM to model temporal dependencies
    in radar pulse characteristics.
    """
    
    def __init__(
        self,
        hidden: int = 256,
        layers: int = 2,
        bidirectional: bool = True,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(out_dim)
        
        self.hidden_size = hidden
        self.num_layers = layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Input dimension will be inferred from first forward pass
        self.input_dim = None
        self.lstm = None
        self.feature_head = None
        self.pulse_analyzer = None
        self.attention_pool = None
        self.event_detectors = None
        
        self._initialized = False
    
    def _initialize_layers(self, input_dim: int) -> None:
        """Initialize layers based on input dimension."""
        self.input_dim = input_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_output_dim)
        
        # Pulse characteristics analyzer
        self.pulse_analyzer = PulseCharacteristics(lstm_output_dim)
        
        # Feature projection head
        self.feature_head = nn.Sequential(
            nn.Linear(lstm_output_dim, self.out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )
        
        # Event detection heads
        self.event_detectors = nn.ModuleDict({
            'pulse_pattern_anomaly': nn.Sequential(
                nn.Linear(self.out_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ),
            'prf_change': nn.Sequential(
                nn.Linear(self.out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ),
            'jamming_detected': nn.Sequential(
                nn.Linear(self.out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ),
        })
        
        self._initialized = True
    
    def forward(self, pulse_feat: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for pulse feature processing.
        
        Args:
            pulse_feat: Pulse features of shape (B, T, D) where:
                       T = time steps, D = feature dimension
                       
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        if not self._initialized:
            self._initialize_layers(pulse_feat.shape[-1])
        
        B, T, D = pulse_feat.shape
        
        # Pack sequences for efficiency (assuming all sequences have same length)
        lstm_out, (h_n, c_n) = self.lstm(pulse_feat)  # (B, T, H)
        
        # Apply attention pooling
        pooled_features = self.attention_pool(lstm_out)  # (B, H)
        
        # Analyze pulse characteristics
        prf_estimate, pulse_width, am_probability = self.pulse_analyzer(pooled_features)
        
        # Project to output features
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
            # Pulse pattern anomaly event
            anomaly_prob = events_output['pulse_pattern_anomaly'][b, 0].item()
            if anomaly_prob > 0.7:
                events.append(EventToken(
                    type="pulse_pattern_anomaly",
                    value=anomaly_prob,
                    t_start=0.0,
                    t_end=T / 1000.0,  # Assume millisecond resolution
                    quality=anomaly_prob,
                    meta={
                        "prf_estimate_hz": max(0, prf_estimate[b, 0].item()),
                        "pulse_width_us": max(0, pulse_width[b, 0].item()),
                        "am_modulation_prob": am_probability[b, 0].item(),
                        "sequence_length": T,
                    }
                ))
            
            # PRF change event
            prf_change = events_output['prf_change'][b, 0].item()
            if abs(prf_change) > 100.0:  # Hz change threshold
                events.append(EventToken(
                    type="prf_change_hz",
                    value=prf_change,
                    t_start=0.0,
                    t_end=T / 1000.0,
                    quality=min(abs(prf_change) / 1000.0, 1.0),
                    meta={
                        "prf_direction": "increase" if prf_change > 0 else "decrease",
                        "magnitude_hz": abs(prf_change),
                        "baseline_prf": prf_estimate[b, 0].item(),
                    }
                ))
            
            # Jamming detection event
            jamming_prob = events_output['jamming_detected'][b, 0].item()
            if jamming_prob > 0.8:
                events.append(EventToken(
                    type="jamming_detected",
                    value=jamming_prob,
                    t_start=0.0,
                    t_end=T / 1000.0,
                    quality=jamming_prob,
                    meta={
                        "jamming_type": "suspected_noise" if am_probability[b, 0].item() > 0.5 else "suspected_coherent",
                        "signal_degradation": anomaly_prob,
                        "prf_stability": 1.0 - abs(prf_change) / 1000.0,
                    }
                ))
        
        # Store intermediate outputs
        self._last_lstm_out = lstm_out
        self._last_attention_weights = self.attention_pool.attention(lstm_out)
        self._last_pulse_analysis = (prf_estimate, pulse_width, am_probability)
        self._last_events_raw = events_output
        
        return feature_vec, events
    
    def get_lstm_features(self) -> torch.Tensor:
        """Get LSTM output features."""
        if hasattr(self, '_last_lstm_out'):
            return self._last_lstm_out
        else:
            raise RuntimeError("No LSTM features available. Run forward() first.")
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get attention weights for sequence visualization."""
        if hasattr(self, '_last_attention_weights'):
            return torch.softmax(self._last_attention_weights, dim=1)
        else:
            raise RuntimeError("No attention weights available. Run forward() first.")
    
    def get_pulse_analysis(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get pulse characteristics analysis results."""
        if hasattr(self, '_last_pulse_analysis'):
            return self._last_pulse_analysis
        else:
            raise RuntimeError("No pulse analysis available. Run forward() first.")
    
    def get_event_scores(self) -> dict:
        """Get raw event detection scores."""
        if hasattr(self, '_last_events_raw'):
            return self._last_events_raw
        else:
            raise RuntimeError("No event scores available. Run forward() first.")


def create_pulse_lstm(config: dict) -> PulseLSTM:
    """Factory function to create PulseLSTM from config."""
    return PulseLSTM(
        hidden=config.get("hidden", 256),
        layers=config.get("layers", 2),
        bidirectional=config.get("bidirectional", True),
        out_dim=config.get("out_dim", 256),
        dropout=config.get("dropout", 0.1),
    )