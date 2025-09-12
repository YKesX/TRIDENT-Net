"""
TRIDENT-T2: CoolCurve3 - 3-point decay fit and MLP classifier over thermal curves

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..common.types import BranchModule, EventToken


class CoolCurve3(BranchModule):
    """
    3-point decay fit + MLP classifier over per-track intensity/area curves.
    
    Analyzes thermal decay patterns from tracked objects to classify
    debris vs flare signatures and estimate cooling time constants.
    
    Input:
        - curves (B, M_max=10, 3) - per-track intensity curves
        - areas (B, M_max=10, 3) - per-track area curves  
        - pad_mask (B, M_max) - valid track mask
    Outputs:
        - tau_hat (B, 1) - estimated cooling time constant
        - debris_vs_flare (B, 2) - classification logits
        - zt (B, 256) - pooled embedding
        - events (list) - cooling curve events
    """
    
    def __init__(self, use_log_linear_fit: bool = True, mlp_hidden: int = 128, out_dim: int = 256):
        super().__init__(out_dim=out_dim)
        
        self.use_log_linear_fit = use_log_linear_fit
        self.max_tracks = 10
        
        # Physics-based feature extractor
        self.physics_extractor = PhysicsFeatureExtractor()
        
        # Per-track curve analyzer
        self.curve_analyzer = CurveAnalyzer(mlp_hidden)
        
        # Track aggregation network
        self.track_aggregator = TrackAggregator(mlp_hidden // 2, out_dim)
        
        # Cooling time constant estimator
        self.tau_estimator = nn.Sequential(
            nn.Linear(out_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive tau
        )
        
        # Debris vs flare classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # debris, flare logits
        )
        
    def forward(self, curves: torch.Tensor, areas: torch.Tensor, 
                pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[EventToken]]:
        """
        Forward pass through CoolCurve3.
        
        Args:
            curves: Intensity curves (B, M_max, 3)
            areas: Area curves (B, M_max, 3)
            pad_mask: Valid track mask (B, M_max)
            
        Returns:
            tuple: (tau_hat, debris_vs_flare, zt, events)
                - tau_hat: (B, 1) cooling time constant
                - debris_vs_flare: (B, 2) classification logits
                - zt: (B, 256) feature embedding
                - events: List of cooling curve events
        """
        B, M, T = curves.shape
        
        # Extract physics-based features for each track
        physics_features = self.physics_extractor(curves, areas)  # (B, M, n_physics_features)
        
        # Analyze each track's cooling curve
        track_features = self.curve_analyzer(curves, areas, physics_features)  # (B, M, feature_dim)
        
        # Apply padding mask
        track_features = track_features * pad_mask.unsqueeze(-1)
        
        # Aggregate across tracks
        zt = self.track_aggregator(track_features, pad_mask)  # (B, out_dim)
        
        # Estimate cooling time constant
        tau_hat = self.tau_estimator(zt)  # (B, 1)
        
        # Classify debris vs flare
        debris_vs_flare = self.classifier(zt)  # (B, 2)
        
        # Extract cooling events
        events = self._extract_cooling_events(curves, areas, pad_mask, tau_hat, debris_vs_flare)
        
        return tau_hat, debris_vs_flare, zt, events
    
    def _extract_cooling_events(self, curves: torch.Tensor, areas: torch.Tensor,
                               pad_mask: torch.Tensor, tau_hat: torch.Tensor, 
                               debris_vs_flare: torch.Tensor) -> List[EventToken]:
        """Extract cooling curve events."""
        events = []
        B, M, T = curves.shape
        
        for b in range(B):
            tau_val = tau_hat[b, 0].item()
            debris_logit, flare_logit = debris_vs_flare[b]
            debris_prob = torch.softmax(debris_vs_flare[b], dim=0)[0].item()
            
            # Find significant tracks
            valid_tracks = pad_mask[b].nonzero(as_tuple=False).squeeze(-1)
            
            for track_idx in valid_tracks:
                track_idx = track_idx.item()
                intensity_curve = curves[b, track_idx]  # (3,)
                area_curve = areas[b, track_idx]  # (3,)
                
                # Check for significant cooling pattern
                if self._is_significant_cooling(intensity_curve, area_curve):
                    # Determine event type
                    if debris_prob > 0.6:
                        event_type = "debris_cooling"
                    elif debris_prob < 0.4:
                        event_type = "flare_cooling"
                    else:
                        event_type = "thermal_decay"
                    
                    # Estimate cooling rate
                    cooling_rate = self._compute_cooling_rate(intensity_curve)
                    
                    event = EventToken(
                        type=event_type,
                        score=max(abs(debris_prob - 0.5) * 2, 0.5),  # Confidence from classification
                        t_ms=200,  # End of sequence, convert to ms
                        meta={
                            'track_id': track_idx,
                            'tau_estimate': tau_val,
                            'debris_probability': debris_prob,
                            'cooling_rate': cooling_rate,
                            'initial_intensity': intensity_curve[0].item(),
                            'final_intensity': intensity_curve[-1].item(),
                            'location': (0, 0),  # No spatial location for cooling curves
                            'source': "coolcurve3",
                            'batch_idx': b
                        }
                    )
                    events.append(event)
        
        return events
    
    def _is_significant_cooling(self, intensity_curve: torch.Tensor, area_curve: torch.Tensor) -> bool:
        """Check if cooling pattern is significant."""
        # Check for monotonic decrease in intensity
        intensity_decreasing = intensity_curve[0] > intensity_curve[1] > intensity_curve[2]
        
        # Check for minimum intensity change
        intensity_drop = (intensity_curve[0] - intensity_curve[-1]) / (intensity_curve[0] + 1e-8)
        significant_drop = intensity_drop > 0.2
        
        return intensity_decreasing and significant_drop
    
    def _compute_cooling_rate(self, intensity_curve: torch.Tensor) -> float:
        """Compute cooling rate from intensity curve."""
        if self.use_log_linear_fit:
            # Log-linear fit: log(I) = -t/tau + const
            log_intensities = torch.log(intensity_curve + 1e-8)
            time_points = torch.tensor([0.0, 1.0, 2.0])
            
            # Simple linear regression
            mean_t = time_points.mean()
            mean_log_i = log_intensities.mean()
            
            numerator = ((time_points - mean_t) * (log_intensities - mean_log_i)).sum()
            denominator = ((time_points - mean_t) ** 2).sum()
            
            if denominator > 1e-8:
                slope = numerator / denominator
                return -slope.item()  # Positive cooling rate
            else:
                return 0.0
        else:
            # Simple difference-based rate
            return ((intensity_curve[0] - intensity_curve[-1]) / 2.0).item()


class PhysicsFeatureExtractor(nn.Module):
    """Extract physics-based features from cooling curves."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, curves: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
        """Extract physics features."""
        B, M, T = curves.shape
        
        features = []
        
        # Intensity features
        intensity_max = curves.max(dim=-1)[0]  # (B, M)
        intensity_min = curves.min(dim=-1)[0]
        intensity_range = intensity_max - intensity_min
        intensity_mean = curves.mean(dim=-1)
        
        # Area features  
        area_max = areas.max(dim=-1)[0]
        area_min = areas.min(dim=-1)[0]
        area_range = area_max - area_min
        area_mean = areas.mean(dim=-1)
        
        # Temporal gradients
        intensity_grad = curves[:, :, 1:] - curves[:, :, :-1]  # (B, M, 2)
        area_grad = areas[:, :, 1:] - areas[:, :, :-1]
        
        intensity_grad_mean = intensity_grad.mean(dim=-1)
        area_grad_mean = area_grad.mean(dim=-1)
        
        # Ratios and normalized features
        intensity_area_ratio = intensity_mean / (area_mean + 1e-8)
        decay_rate = (curves[:, :, 0] - curves[:, :, -1]) / (curves[:, :, 0] + 1e-8)
        
        # Stack all features
        features = torch.stack([
            intensity_max, intensity_min, intensity_range, intensity_mean,
            area_max, area_min, area_range, area_mean,
            intensity_grad_mean, area_grad_mean,
            intensity_area_ratio, decay_rate
        ], dim=-1)  # (B, M, 12)
        
        return features


class CurveAnalyzer(nn.Module):
    """Analyze individual cooling curves."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Input: curves (3) + areas (3) + physics features (12) = 18
        self.input_dim = 18
        
        self.analyzer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
    def forward(self, curves: torch.Tensor, areas: torch.Tensor, 
                physics_features: torch.Tensor) -> torch.Tensor:
        """Analyze cooling curves."""
        B, M, T = curves.shape
        
        # Concatenate all inputs
        inputs = torch.cat([curves, areas, physics_features], dim=-1)  # (B, M, 18)
        
        # Reshape for batch processing
        inputs_flat = inputs.view(B * M, -1)  # (B*M, 18)
        
        # Analyze
        outputs_flat = self.analyzer(inputs_flat)  # (B*M, hidden_dim//2)
        
        # Reshape back
        outputs = outputs_flat.view(B, M, -1)  # (B, M, hidden_dim//2)
        
        return outputs


class TrackAggregator(nn.Module):
    """Aggregate features across tracks."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Attention-based aggregation
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, track_features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate track features with attention."""
        B, M, D = track_features.shape
        
        # Compute attention weights
        attention_logits = self.attention(track_features)  # (B, M, 1)
        attention_logits = attention_logits.squeeze(-1)  # (B, M)
        
        # Apply mask (set masked positions to large negative value)
        attention_logits = attention_logits.masked_fill(~pad_mask.bool(), -1e9)
        
        # Softmax attention
        attention_weights = F.softmax(attention_logits, dim=-1)  # (B, M)
        
        # Weighted aggregation
        aggregated = (track_features * attention_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        # Final projection
        output = self.output_proj(aggregated)
        
        return output