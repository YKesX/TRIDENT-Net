"""
PulseLSTM: Bidirectional LSTM for pulse feature processing.

Processes radar pulse sequences using bidirectional LSTM layers
for temporal modeling of radar returns and target characteristics.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken, FeatureVec


class PulseLSTM(BranchModule):
    """
    Pulse LSTM for radar pulse sequence processing.
    
    Uses bidirectional LSTM layers to model temporal dependencies
    in radar pulse features and detect target-related events.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        out_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ) -> None:
        """
        Initialize PulseLSTM.
        
        Args:
            input_dim: Input pulse feature dimension
            out_dim: Output feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Determine LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism for sequence aggregation
        self.attention = PulseAttentionModule(
            input_dim=lstm_output_dim,
            attention_dim=128
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim)
        )
        
        # Event detection heads
        self.range_gate_detector = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.amplitude_anomaly_detector = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.phase_shift_detector = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Pulse timing analyzer
        self.timing_analyzer = PulseTimingAnalyzer(lstm_output_dim)
    
    def forward(self, pulse_feat: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process radar pulse features.
        
        Args:
            pulse_feat: Pulse features (B, T, F) where T=pulses, F=features
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size, seq_len, _ = pulse_feat.shape
        
        # Input projection
        x = self.input_proj(pulse_feat)
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Apply attention for sequence aggregation
        attended_features, attention_weights = self.attention(lstm_output)
        
        # Feature projection
        projected_features = self.feature_proj(attended_features)
        
        # Event detection
        range_gate_score = self.range_gate_detector(attended_features)
        amplitude_score = self.amplitude_anomaly_detector(attended_features)
        phase_score = self.phase_shift_detector(attended_features)
        
        # Timing analysis
        timing_events = self.timing_analyzer(lstm_output, attention_weights)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Generate event tokens
        events = []
        for b in range(batch_size):
            # Range gate change detection
            if range_gate_score[b] > 0.7:
                # Find time of range gate change
                peak_time = torch.argmax(attention_weights[b]).item()
                
                events.append(EventToken(
                    type="range_gate_change",
                    value=range_gate_score[b].item(),
                    t_start=peak_time - 1.0,
                    t_end=peak_time + 1.0,
                    quality=range_gate_score[b].item(),
                    meta={
                        "source": "pulse_lstm",
                        "batch_idx": b,
                        "peak_time": peak_time
                    }
                ))
            
            # Amplitude anomaly detection
            if amplitude_score[b] > 0.6:
                events.append(EventToken(
                    type="amplitude_anomaly",
                    value=amplitude_score[b].item(),
                    t_start=0.0,
                    t_end=seq_len,
                    quality=amplitude_score[b].item(),
                    meta={
                        "source": "pulse_lstm",
                        "batch_idx": b,
                        "anomaly_strength": amplitude_score[b].item()
                    }
                ))
            
            # Phase shift detection
            if phase_score[b] > 0.5:
                events.append(EventToken(
                    type="phase_shift",
                    value=phase_score[b].item(),
                    t_start=0.0,
                    t_end=seq_len,
                    quality=phase_score[b].item(),
                    meta={
                        "source": "pulse_lstm",
                        "batch_idx": b,
                        "phase_coherence": 1.0 - phase_score[b].item()
                    }
                ))
            
            # Add timing-based events
            for timing_event in timing_events:
                if timing_event["batch_idx"] == b:
                    events.append(EventToken(
                        type=timing_event["type"],
                        value=timing_event["value"],
                        t_start=timing_event["t_start"],
                        t_end=timing_event["t_end"],
                        quality=timing_event["quality"],
                        meta=timing_event["meta"]
                    ))
        
        return feature_vec, events
    
    def get_hidden_states(self, pulse_feat: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM hidden states for analysis.
        
        Args:
            pulse_feat: Pulse features (B, T, F)
            
        Returns:
            LSTM hidden states (B, T, hidden_dim * directions)
        """
        x = self.input_proj(pulse_feat)
        lstm_output, _ = self.lstm(x)
        return lstm_output


class PulseAttentionModule(nn.Module):
    """Attention mechanism for pulse sequence aggregation."""
    
    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128
    ) -> None:
        """
        Initialize pulse attention module.
        
        Args:
            input_dim: Input feature dimension
            attention_dim: Attention computation dimension
        """
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.Tanh(),
            nn.Linear(attention_dim // 2, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self, 
        sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to sequence.
        
        Args:
            sequence: Input sequence (B, T, C)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention_net(sequence)  # (B, T, 1)
        attention_weights = self.softmax(attention_scores.squeeze(-1))  # (B, T)
        
        # Apply attention
        attended_features = torch.sum(
            sequence * attention_weights.unsqueeze(-1), dim=1
        )  # (B, C)
        
        return attended_features, attention_weights


class PulseTimingAnalyzer(nn.Module):
    """Analyzer for pulse timing irregularities."""
    
    def __init__(self, feature_dim: int) -> None:
        """
        Initialize pulse timing analyzer.
        
        Args:
            feature_dim: Input feature dimension
        """
        super().__init__()
        
        self.timing_estimator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.jitter_detector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        lstm_output: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> List[dict]:
        """
        Analyze pulse timing for irregularities.
        
        Args:
            lstm_output: LSTM output sequence (B, T, C)
            attention_weights: Attention weights (B, T)
            
        Returns:
            List of timing events
        """
        batch_size, seq_len, _ = lstm_output.shape
        events = []
        
        for b in range(batch_size):
            # Analyze timing for this batch
            sequence_features = lstm_output[b]  # (T, C)
            weights = attention_weights[b]  # (T,)
            
            # Compute timing metrics
            timing_scores = self.timing_estimator(sequence_features).squeeze(-1)  # (T,)
            jitter_scores = self.jitter_detector(sequence_features).squeeze(-1)  # (T,)
            
            # Detect timing anomalies
            timing_threshold = 0.7
            jitter_threshold = 0.6
            
            # Find periods of timing irregularity
            timing_anomalies = timing_scores > timing_threshold
            jitter_anomalies = jitter_scores > jitter_threshold
            
            if timing_anomalies.any():
                anomaly_indices = torch.where(timing_anomalies)[0]
                start_time = anomaly_indices[0].item()
                end_time = anomaly_indices[-1].item() + 1
                
                events.append({
                    "type": "timing_irregularity",
                    "value": timing_scores[timing_anomalies].mean().item(),
                    "t_start": start_time,
                    "t_end": end_time,
                    "quality": timing_scores[timing_anomalies].mean().item(),
                    "batch_idx": b,
                    "meta": {
                        "source": "pulse_timing_analyzer",
                        "anomaly_duration": end_time - start_time,
                        "affected_pulses": len(anomaly_indices)
                    }
                })
            
            if jitter_anomalies.any():
                jitter_indices = torch.where(jitter_anomalies)[0]
                start_time = jitter_indices[0].item()
                end_time = jitter_indices[-1].item() + 1
                
                events.append({
                    "type": "pulse_jitter",
                    "value": jitter_scores[jitter_anomalies].mean().item(),
                    "t_start": start_time,
                    "t_end": end_time,
                    "quality": jitter_scores[jitter_anomalies].mean().item(),
                    "batch_idx": b,
                    "meta": {
                        "source": "pulse_timing_analyzer",
                        "jitter_severity": jitter_scores[jitter_anomalies].max().item(),
                        "affected_pulses": len(jitter_indices)
                    }
                })
        
        return events


class PulseFeatureExtractor(nn.Module):
    """Extract features from raw pulse data."""
    
    def __init__(
        self,
        raw_pulse_dim: int,
        feature_dim: int = 64
    ) -> None:
        """
        Initialize pulse feature extractor.
        
        Args:
            raw_pulse_dim: Raw pulse data dimension
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(raw_pulse_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim)
        )
        
        # Specialized extractors for different aspects
        self.amplitude_extractor = nn.Linear(raw_pulse_dim, 16)
        self.phase_extractor = nn.Linear(raw_pulse_dim, 16)
        self.timing_extractor = nn.Linear(raw_pulse_dim, 16)
        self.doppler_extractor = nn.Linear(raw_pulse_dim, 16)
    
    def forward(self, raw_pulse: torch.Tensor) -> torch.Tensor:
        """
        Extract features from raw pulse data.
        
        Args:
            raw_pulse: Raw pulse data (B, T, raw_dim)
            
        Returns:
            Extracted features (B, T, feature_dim)
        """
        # Main feature extraction
        main_features = self.feature_extractor(raw_pulse)
        
        # Specialized feature extraction
        amplitude_feat = self.amplitude_extractor(raw_pulse)
        phase_feat = self.phase_extractor(raw_pulse)
        timing_feat = self.timing_extractor(raw_pulse)
        doppler_feat = self.doppler_extractor(raw_pulse)
        
        # Combine features
        combined_features = torch.cat([
            main_features,
            amplitude_feat,
            phase_feat,
            timing_feat,
            doppler_feat
        ], dim=-1)
        
        return combined_features


# Default configuration
PULSE_LSTM_CONFIG = {
    "input_dim": 64,
    "out_dim": 256,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True
}