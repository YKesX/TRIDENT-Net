"""
CoolingCurve: Temporal classifier for thermal intensity analysis.

Uses GRU/MLP to analyze thermal intensity curves over time
and estimate cooling parameters and thermal classification.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken, FeatureVec


class CoolingCurve(BranchModule):
    """
    Cooling curve analyzer for thermal temporal patterns.
    
    Uses GRU/MLP architecture to model thermal intensity curves
    and estimate cooling time constants and thermal classification.
    """
    
    def __init__(
        self,
        input_dim: int = 1,  # Single intensity value per time step
        out_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,  # thermal classification types
        dropout: float = 0.2
    ) -> None:
        """
        Initialize CoolingCurve.
        
        Args:
            input_dim: Input dimension (intensity values)
            out_dim: Output feature dimension
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            num_classes: Number of thermal classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input preprocessing
        self.input_preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Determine GRU output dimension
        gru_output_dim = hidden_dim * 2  # Bidirectional
        
        # Temporal attention for curve analysis
        self.temporal_attention = TemporalCurveAttention(gru_output_dim)
        
        # Curve parameter estimation
        self.tau_regressor = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive tau values
        )
        
        # Thermal classification
        self.thermal_classifier = nn.Sequential(
            nn.Linear(gru_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(gru_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim)
        )
        
        # Curve quality assessor
        self.curve_quality = CurveQualityAssessor(gru_output_dim)
        
        # Cooling phase detector
        self.cooling_detector = CoolingPhaseDetector(gru_output_dim)
    
    def forward(self, ir_roi_t: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Analyze thermal intensity curves.
        
        Args:
            ir_roi_t: IR temporal sequence (B, T, 1, H, W)
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size, seq_len, channels, height, width = ir_roi_t.shape
        
        # Extract intensity curves (spatial average per frame)
        intensity_curves = torch.mean(ir_roi_t, dim=[2, 3, 4])  # (B, T)
        
        # Preprocess intensity curves
        preprocessed = self.input_preprocess(intensity_curves.unsqueeze(-1))  # (B, T, hidden_dim//2)
        
        # GRU processing
        gru_output, _ = self.gru(preprocessed)  # (B, T, hidden_dim*2)
        
        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(gru_output)
        
        # Estimate cooling time constant (tau)
        tau_estimate = self.tau_regressor(attended_features)
        
        # Thermal classification
        class_logits = self.thermal_classifier(attended_features)
        class_probs = torch.softmax(class_logits, dim=-1)
        
        # Feature projection
        projected_features = self.feature_proj(attended_features)
        
        # Curve quality assessment
        quality_metrics = self.curve_quality(gru_output, intensity_curves)
        
        # Cooling phase detection
        cooling_phases = self.cooling_detector(gru_output, intensity_curves)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Generate event tokens
        events = []
        thermal_classes = ["rapid_cooling", "slow_cooling", "steady_state", "heating"]
        
        for b in range(batch_size):
            # Tau regression event
            tau_value = tau_estimate[b].item()
            if tau_value > 0.1:  # Significant cooling time constant
                events.append(EventToken(
                    type="cooling_tau",
                    value=tau_value,
                    t_start=0.0,
                    t_end=seq_len,
                    quality=quality_metrics[b]["tau_confidence"],
                    meta={
                        "source": "cooling_curve",
                        "batch_idx": b,
                        "tau_seconds": tau_value,
                        "curve_r_squared": quality_metrics[b]["r_squared"]
                    }
                ))
            
            # Thermal classification event
            max_class_idx = torch.argmax(class_probs[b]).item()
            max_class_prob = class_probs[b, max_class_idx].item()
            
            if max_class_prob > 0.6:
                events.append(EventToken(
                    type=f"thermal_{thermal_classes[max_class_idx]}",
                    value=max_class_prob,
                    t_start=0.0,
                    t_end=seq_len,
                    quality=max_class_prob,
                    meta={
                        "source": "cooling_curve",
                        "batch_idx": b,
                        "thermal_class": thermal_classes[max_class_idx],
                        "class_probabilities": class_probs[b].tolist()
                    }
                ))
            
            # Cooling phase events
            for phase in cooling_phases[b]:
                events.append(EventToken(
                    type=phase["type"],
                    value=phase["intensity"],
                    t_start=phase["start_time"],
                    t_end=phase["end_time"],
                    quality=phase["confidence"],
                    meta={
                        "source": "cooling_curve",
                        "batch_idx": b,
                        **phase["meta"]
                    }
                ))
        
        return feature_vec, events
    
    def get_intensity_curves(self, ir_roi_t: torch.Tensor) -> torch.Tensor:
        """
        Extract intensity curves for visualization.
        
        Args:
            ir_roi_t: IR temporal sequence (B, T, 1, H, W)
            
        Returns:
            Intensity curves (B, T)
        """
        return torch.mean(ir_roi_t, dim=[2, 3, 4])
    
    def fit_cooling_model(
        self,
        intensity_curve: torch.Tensor,
        time_steps: torch.Tensor
    ) -> dict:
        """
        Fit exponential cooling model to intensity curve.
        
        Args:
            intensity_curve: Intensity values (T,)
            time_steps: Time values (T,)
            
        Returns:
            Fitted model parameters
        """
        # Simple exponential decay fitting
        # I(t) = I0 * exp(-t/tau) + I_inf
        
        # Use log-linear regression for initial estimate
        log_intensity = torch.log(intensity_curve + 1e-8)
        
        # Fit linear model: log(I) = log(I0) - t/tau
        X = torch.stack([torch.ones_like(time_steps), time_steps], dim=1)
        y = log_intensity
        
        # Least squares solution
        try:
            coeffs = torch.linalg.lstsq(X, y).solution
            log_I0, neg_inv_tau = coeffs
            
            I0 = torch.exp(log_I0)
            tau = -1.0 / neg_inv_tau if neg_inv_tau != 0 else torch.tensor(float('inf'))
            
            # Compute R-squared
            y_pred = X @ coeffs
            ss_res = torch.sum((y - y_pred) ** 2)
            ss_tot = torch.sum((y - torch.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else torch.tensor(0.0)
            
            return {
                "I0": I0.item(),
                "tau": tau.item(),
                "r_squared": r_squared.item()
            }
        
        except:
            return {
                "I0": intensity_curve[0].item(),
                "tau": 1.0,
                "r_squared": 0.0
            }


class TemporalCurveAttention(nn.Module):
    """Attention mechanism specialized for thermal curves."""
    
    def __init__(self, input_dim: int, attention_dim: int = 128) -> None:
        """
        Initialize temporal curve attention.
        
        Args:
            input_dim: Input feature dimension
            attention_dim: Attention computation dimension
        """
        super().__init__()
        
        # Attention computation with curve-specific features
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.Tanh(),
            nn.Linear(attention_dim // 2, 1)
        )
        
        # Temperature derivative analyzer
        self.derivative_analyzer = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self, 
        sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply curve-aware attention.
        
        Args:
            sequence: Input sequence (B, T, C)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Base attention scores
        attention_scores = self.attention_net(sequence)  # (B, T, 1)
        
        # Compute attention weights
        attention_weights = self.softmax(attention_scores.squeeze(-1))  # (B, T)
        
        # Apply attention
        attended_features = torch.sum(
            sequence * attention_weights.unsqueeze(-1), dim=1
        )  # (B, C)
        
        return attended_features, attention_weights


class CurveQualityAssessor(nn.Module):
    """Assess quality and fit of thermal curves."""
    
    def __init__(self, feature_dim: int) -> None:
        """
        Initialize curve quality assessor.
        
        Args:
            feature_dim: Feature dimension
        """
        super().__init__()
        
        self.quality_estimator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # [tau_confidence, r_squared, noise_level]
            nn.Sigmoid()
        )
        
        self.smoothness_analyzer = nn.Conv1d(1, 1, kernel_size=5, padding=2)
    
    def forward(
        self,
        gru_output: torch.Tensor,
        intensity_curves: torch.Tensor
    ) -> List[dict]:
        """
        Assess curve quality metrics.
        
        Args:
            gru_output: GRU output (B, T, C)
            intensity_curves: Intensity curves (B, T)
            
        Returns:
            List of quality metrics per batch
        """
        batch_size = gru_output.shape[0]
        
        # Global features for quality assessment
        global_features = torch.mean(gru_output, dim=1)  # (B, C)
        quality_scores = self.quality_estimator(global_features)  # (B, 3)
        
        # Analyze curve smoothness
        smoothness_scores = []
        for b in range(batch_size):
            curve = intensity_curves[b:b+1].unsqueeze(1)  # (1, 1, T)
            smoothed = self.smoothness_analyzer(curve)
            smoothness = 1.0 - torch.mean(torch.abs(curve - smoothed)).item()
            smoothness_scores.append(max(0.0, smoothness))
        
        # Compile quality metrics
        quality_metrics = []
        for b in range(batch_size):
            quality_metrics.append({
                "tau_confidence": quality_scores[b, 0].item(),
                "r_squared": quality_scores[b, 1].item(),
                "noise_level": quality_scores[b, 2].item(),
                "smoothness": smoothness_scores[b]
            })
        
        return quality_metrics


class CoolingPhaseDetector(nn.Module):
    """Detect different phases in thermal cooling curves."""
    
    def __init__(self, feature_dim: int) -> None:
        """
        Initialize cooling phase detector.
        
        Args:
            feature_dim: Feature dimension
        """
        super().__init__()
        
        self.phase_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # [rapid_cooling, slow_cooling, plateau, heating]
            nn.Softmax(dim=-1)
        )
        
        # Phase transition detector
        self.transition_detector = nn.Conv1d(
            feature_dim, 1, kernel_size=3, padding=1
        )
    
    def forward(
        self,
        gru_output: torch.Tensor,
        intensity_curves: torch.Tensor
    ) -> List[List[dict]]:
        """
        Detect cooling phases in thermal curves.
        
        Args:
            gru_output: GRU output (B, T, C)
            intensity_curves: Intensity curves (B, T)
            
        Returns:
            List of phase events per batch
        """
        batch_size, seq_len, feature_dim = gru_output.shape
        
        # Classify each time step
        time_step_features = gru_output.view(-1, feature_dim)  # (B*T, C)
        phase_probs = self.phase_classifier(time_step_features)  # (B*T, 4)
        phase_probs = phase_probs.view(batch_size, seq_len, 4)  # (B, T, 4)
        
        # Detect phase transitions
        gru_transposed = gru_output.transpose(1, 2)  # (B, C, T)
        transition_scores = torch.sigmoid(
            self.transition_detector(gru_transposed)
        ).squeeze(1)  # (B, T)
        
        phase_names = ["rapid_cooling", "slow_cooling", "plateau", "heating"]
        all_phase_events = []
        
        for b in range(batch_size):
            batch_phases = []
            
            # Find dominant phases
            dominant_phases = torch.argmax(phase_probs[b], dim=1)  # (T,)
            transitions = transition_scores[b] > 0.7  # (T,)
            
            # Segment into continuous phases
            current_phase = dominant_phases[0].item()
            phase_start = 0
            
            for t in range(1, seq_len):
                if dominant_phases[t] != current_phase or transitions[t]:
                    # End current phase
                    phase_intensity = torch.mean(
                        intensity_curves[b, phase_start:t]
                    ).item()
                    phase_confidence = torch.mean(
                        phase_probs[b, phase_start:t, current_phase]
                    ).item()
                    
                    batch_phases.append({
                        "type": f"cooling_phase_{phase_names[current_phase]}",
                        "intensity": phase_intensity,
                        "start_time": phase_start,
                        "end_time": t,
                        "confidence": phase_confidence,
                        "meta": {
                            "phase_name": phase_names[current_phase],
                            "phase_duration": t - phase_start,
                            "average_intensity": phase_intensity
                        }
                    })
                    
                    # Start new phase
                    current_phase = dominant_phases[t].item()
                    phase_start = t
            
            # Handle final phase
            if phase_start < seq_len - 1:
                phase_intensity = torch.mean(
                    intensity_curves[b, phase_start:]
                ).item()
                phase_confidence = torch.mean(
                    phase_probs[b, phase_start:, current_phase]
                ).item()
                
                batch_phases.append({
                    "type": f"cooling_phase_{phase_names[current_phase]}",
                    "intensity": phase_intensity,
                    "start_time": phase_start,
                    "end_time": seq_len,
                    "confidence": phase_confidence,
                    "meta": {
                        "phase_name": phase_names[current_phase],
                        "phase_duration": seq_len - phase_start,
                        "average_intensity": phase_intensity
                    }
                })
            
            all_phase_events.append(batch_phases)
        
        return all_phase_events


# Default configuration
COOLING_CURVE_CONFIG = {
    "input_dim": 1,
    "out_dim": 256,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_classes": 4,
    "dropout": 0.2
}