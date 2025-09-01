"""
TRIDENT-T2: Cooling Curve Analysis (temporal GRU/MLP)

Author: Yağızhan Keskin
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..common.types import BranchModule, EventToken, FeatureVec


class CoolingCurveAnalyzer(nn.Module):
    """Analyze thermal cooling curve characteristics."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Exponential fitting parameters estimation
        self.tau_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive tau
        )
        
        # Initial temperature estimation
        self.t0_estimator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        # Curve quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze cooling curve parameters.
        
        Args:
            features: GRU features (B, H)
            
        Returns:
            tuple: (tau, t0, quality)
        """
        tau = self.tau_estimator(features)  # Time constant
        t0 = self.t0_estimator(features)    # Initial temperature
        quality = self.quality_assessor(features)  # Fit quality
        
        return tau, t0, quality


class DebrisFlareClassifier(nn.Module):
    """Classify thermal signature as debris vs flare."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # debris, flare
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify thermal signature type.
        
        Args:
            features: GRU features (B, H)
            
        Returns:
            Class logits (B, 2)
        """
        return self.classifier(features)


class CoolingCurve(BranchModule):
    """
    Cooling Curve Analysis module for thermal temporal classification.
    
    Uses GRU to model temporal dependencies in thermal cooling patterns
    and estimates physical parameters like time constants.
    """
    
    def __init__(
        self,
        arch: str = "gru",
        hidden: int = 128,
        layers: int = 2,
        out_dim: int = 256,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__(out_dim)
        
        self.arch = arch
        self.hidden_size = hidden
        self.num_layers = layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Input dimension will be inferred
        self.input_dim = None
        self.rnn = None
        self.cooling_analyzer = None
        self.debris_classifier = None
        self.feature_head = None
        self.event_detectors = None
        
        self._initialized = False
    
    def _initialize_layers(self, input_dim: int) -> None:
        """Initialize layers based on input dimension."""
        self.input_dim = input_dim
        
        # RNN layer
        if self.arch == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
            )
        elif self.arch == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
            )
        else:
            raise ValueError(f"Unknown RNN architecture: {self.arch}")
        
        # Calculate RNN output dimension
        rnn_output_dim = self.hidden_size * (2 if self.bidirectional else 1)
        
        # Cooling curve analyzer
        self.cooling_analyzer = CoolingCurveAnalyzer(rnn_output_dim)
        
        # Debris vs flare classifier
        self.debris_classifier = DebrisFlareClassifier(rnn_output_dim)
        
        # Feature projection head
        self.feature_head = nn.Sequential(
            nn.Linear(rnn_output_dim, self.out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )
        
        # Event detection heads
        self.event_detectors = nn.ModuleDict({
            'cooling_rate_anomaly': nn.Sequential(
                nn.Linear(self.out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ),
            'temperature_spike': nn.Sequential(
                nn.Linear(self.out_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
            'thermal_persistence': nn.Sequential(
                nn.Linear(self.out_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            ),
        })
        
        self._initialized = True
    
    def forward(self, curve_seq: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for cooling curve analysis.
        
        Args:
            curve_seq: Cooling curve sequence of shape (B, T, D) where:
                      T = time steps, D = curve features (temperature, intensity, etc.)
                      
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        if not self._initialized:
            self._initialize_layers(curve_seq.shape[-1])
        
        B, T, D = curve_seq.shape
        
        # Apply RNN
        if self.arch == "gru":
            rnn_out, hidden = self.rnn(curve_seq)  # (B, T, H)
        else:  # LSTM
            rnn_out, (hidden, cell) = self.rnn(curve_seq)
        
        # Use last timestep for analysis (or mean pooling)
        last_output = rnn_out[:, -1, :]  # (B, H)
        mean_output = rnn_out.mean(dim=1)   # (B, H)
        combined_output = (last_output + mean_output) / 2
        
        # Cooling curve analysis
        tau, t0, fit_quality = self.cooling_analyzer(combined_output)
        
        # Debris vs flare classification
        debris_flare_logits = self.debris_classifier(combined_output)
        debris_flare_probs = F.softmax(debris_flare_logits, dim=1)
        
        # Project to output features
        global_features = self.feature_head(combined_output)  # (B, out_dim)
        
        # Event detection
        events_output = {}
        for event_type, detector in self.event_detectors.items():
            events_output[event_type] = detector(global_features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=global_features)
        
        # Create EventTokens
        events = []
        for b in range(B):
            # Cooling rate analysis event
            tau_val = tau[b, 0].item()
            fit_qual = fit_quality[b, 0].item()
            
            if fit_qual > 0.7 and tau_val > 0.1:  # Good fit and meaningful time constant
                events.append(EventToken(
                    type="cooling_tau_estimate",
                    value=tau_val,
                    t_start=0.0,
                    t_end=T / 30.0,  # Assume 30 FPS
                    quality=fit_qual,
                    meta={
                        "tau_seconds": tau_val,
                        "initial_temp": t0[b, 0].item(),
                        "fit_quality": fit_qual,
                        "cooling_category": self._categorize_cooling_rate(tau_val),
                    }
                ))
            
            # Debris vs flare classification event
            debris_prob = debris_flare_probs[b, 0].item()
            flare_prob = debris_flare_probs[b, 1].item()
            
            predicted_class = "debris" if debris_prob > flare_prob else "flare"
            confidence = max(debris_prob, flare_prob)
            
            if confidence > 0.7:
                events.append(EventToken(
                    type=f"thermal_{predicted_class}",
                    value=confidence,
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=confidence,
                    meta={
                        "classification": predicted_class,
                        "debris_probability": debris_prob,
                        "flare_probability": flare_prob,
                        "tau_support": tau_val,
                    }
                ))
            
            # Cooling rate anomaly event
            cooling_anomaly = events_output['cooling_rate_anomaly'][b, 0].item()
            if cooling_anomaly > 0.8:
                events.append(EventToken(
                    type="cooling_anomaly",
                    value=cooling_anomaly,
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=cooling_anomaly,
                    meta={
                        "anomaly_type": "irregular_cooling",
                        "expected_tau_range": [0.1, 2.0],
                        "observed_tau": tau_val,
                    }
                ))
            
            # Temperature spike event
            temp_spike = events_output['temperature_spike'][b, 0].item()
            if temp_spike > 1.0:  # Significant spike
                events.append(EventToken(
                    type="temperature_spike",
                    value=temp_spike,
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=min(temp_spike / 5.0, 1.0),
                    meta={
                        "spike_magnitude": temp_spike,
                        "baseline_temp": t0[b, 0].item(),
                    }
                ))
            
            # Thermal persistence event
            persistence = events_output['thermal_persistence'][b, 0].item()
            if persistence > 0.8:
                events.append(EventToken(
                    type="thermal_persistence",
                    value=persistence,
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=persistence,
                    meta={
                        "persistence_level": persistence,
                        "cooling_time_constant": tau_val,
                        "persistence_category": "high" if persistence > 0.9 else "moderate",
                    }
                ))
        
        # Store intermediate outputs
        self._last_rnn_out = rnn_out
        self._last_cooling_params = (tau, t0, fit_quality)
        self._last_classification = debris_flare_probs
        self._last_events_raw = events_output
        
        return feature_vec, events
    
    def _categorize_cooling_rate(self, tau: float) -> str:
        """Categorize cooling rate based on time constant."""
        if tau < 0.2:
            return "very_fast"
        elif tau < 0.5:
            return "fast"
        elif tau < 1.0:
            return "moderate"
        elif tau < 2.0:
            return "slow"
        else:
            return "very_slow"
    
    def get_rnn_features(self) -> torch.Tensor:
        """Get RNN output features."""
        if hasattr(self, '_last_rnn_out'):
            return self._last_rnn_out
        else:
            raise RuntimeError("No RNN features available. Run forward() first.")
    
    def get_cooling_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get cooling curve fit parameters."""
        if hasattr(self, '_last_cooling_params'):
            return self._last_cooling_params
        else:
            raise RuntimeError("No cooling parameters available. Run forward() first.")
    
    def get_classification_results(self) -> torch.Tensor:
        """Get debris vs flare classification probabilities."""
        if hasattr(self, '_last_classification'):
            return self._last_classification
        else:
            raise RuntimeError("No classification results available. Run forward() first.")
    
    def get_event_scores(self) -> dict:
        """Get raw event detection scores."""
        if hasattr(self, '_last_events_raw'):
            return self._last_events_raw
        else:
            raise RuntimeError("No event scores available. Run forward() first.")


def create_cooling_curve(config: dict) -> CoolingCurve:
    """Factory function to create CoolingCurve from config."""
    return CoolingCurve(
        arch=config.get("arch", "gru"),
        hidden=config.get("hidden", 128),
        layers=config.get("layers", 2),
        out_dim=config.get("out_dim", 256),
        dropout=config.get("dropout", 0.1),
        bidirectional=config.get("bidirectional", False),
    )