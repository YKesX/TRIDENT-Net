"""
EchoNet: 1D CNN for micro-doppler sequence processing.

Processes time/frequency micro-doppler sequences using 1D convolutional layers
with temporal attention for radar target classification.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken, FeatureVec


class EchoNet(BranchModule):
    """
    Echo network for micro-doppler sequence analysis.
    
    Uses 1D CNN layers to process temporal radar sequences
    and extract target characteristics from micro-doppler signatures.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        out_dim: int = 256,
        conv_channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [7, 5, 3, 3],
        dropout: float = 0.2
    ) -> None:
        """
        Initialize EchoNet.
        
        Args:
            input_dim: Input frequency bins dimension
            out_dim: Output feature dimension
            conv_channels: Convolutional layer channel sizes
            kernel_sizes: Kernel sizes for each conv layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        # 1D Convolutional layers for micro-doppler processing
        self.conv_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, 
                        kernel_size=kernel_size, 
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout1d(dropout),
                    nn.MaxPool1d(2)
                )
            )
            in_channels = out_channels
        
        # Temporal attention module
        self.temporal_attention = TemporalAttentionModule(
            feature_dim=conv_channels[-1],
            attention_dim=128
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(conv_channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim)
        )
        
        # Event detection heads
        self.rcs_detector = nn.Sequential(
            nn.Linear(conv_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.decorrelation_detector = nn.Sequential(
            nn.Linear(conv_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.velocity_estimator = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()  # Velocity can be positive or negative
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, rd_seq: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process radar micro-doppler sequence.
        
        Args:
            rd_seq: Radar sequence (B, T, F) where T=time, F=frequency
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size, seq_len, freq_bins = rd_seq.shape
        
        # Transpose for 1D conv: (B, F, T)
        x = rd_seq.transpose(1, 2)
        
        # Process through conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(x)
        
        # Global pooling
        pooled_features = self.global_pool(attended_features).squeeze(-1)
        
        # Feature projection
        projected_features = self.feature_proj(pooled_features)
        
        # Event detection
        rcs_score = self.rcs_detector(pooled_features)
        decorr_score = self.decorrelation_detector(pooled_features)
        velocity_est = self.velocity_estimator(pooled_features)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Generate event tokens
        events = []
        for b in range(batch_size):
            # RCS drop detection
            if rcs_score[b] > 0.6:
                # Find time of RCS drop using attention weights
                drop_time = torch.argmax(attention_weights[b]).item()
                
                events.append(EventToken(
                    type="rcs_drop_db",
                    value=rcs_score[b].item() * 10.0,  # Convert to dB scale
                    t_start=drop_time - 1.0,
                    t_end=drop_time + 1.0,
                    quality=rcs_score[b].item(),
                    meta={
                        "source": "echo_net",
                        "batch_idx": b,
                        "drop_time": drop_time
                    }
                ))
            
            # Decorrelation detection
            if decorr_score[b] > 0.5:
                events.append(EventToken(
                    type="decorrelation",
                    value=decorr_score[b].item(),
                    t_start=0.0,
                    t_end=seq_len,
                    quality=decorr_score[b].item(),
                    meta={
                        "source": "echo_net",
                        "batch_idx": b,
                        "velocity_estimate": velocity_est[b].item()
                    }
                ))
            
            # Velocity event (if significant motion detected)
            if abs(velocity_est[b].item()) > 0.3:
                events.append(EventToken(
                    type="velocity_change",
                    value=velocity_est[b].item(),
                    t_start=0.0,
                    t_end=seq_len,
                    quality=min(1.0, abs(velocity_est[b].item()) * 2),
                    meta={
                        "source": "echo_net",
                        "batch_idx": b,
                        "velocity_mps": velocity_est[b].item() * 100  # Scale to m/s
                    }
                ))
        
        return feature_vec, events
    
    def get_doppler_spectrum(self, rd_seq: torch.Tensor) -> torch.Tensor:
        """
        Get processed doppler spectrum for visualization.
        
        Args:
            rd_seq: Radar sequence (B, T, F)
            
        Returns:
            Processed spectrum (B, C, T_reduced)
        """
        # Transpose for conv processing
        x = rd_seq.transpose(1, 2)
        
        # Process through conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        return x


class TemporalAttentionModule(nn.Module):
    """Temporal attention for radar sequence processing."""
    
    def __init__(
        self,
        feature_dim: int,
        attention_dim: int = 128
    ) -> None:
        """
        Initialize temporal attention module.
        
        Args:
            feature_dim: Input feature dimension
            attention_dim: Attention computation dimension
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention computation
        self.attention_conv = nn.Sequential(
            nn.Conv1d(feature_dim, attention_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_dim, attention_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention to features.
        
        Args:
            features: Input features (B, C, T)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Compute attention weights
        attention_weights = self.attention_conv(features)  # (B, 1, T)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Refine features
        refined_features = self.feature_refine(attended_features)
        
        return refined_features, attention_weights.squeeze(1)


class MicroDopplerProcessor(nn.Module):
    """Specialized processor for micro-doppler signatures."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int
    ) -> None:
        """
        Initialize micro-doppler processor.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        # Multi-scale processing
        self.scale_processors = nn.ModuleList([
            nn.Conv1d(input_dim, output_dim // 3, kernel_size=3, padding=1),
            nn.Conv1d(input_dim, output_dim // 3, kernel_size=5, padding=2),
            nn.Conv1d(input_dim, output_dim // 3, kernel_size=7, padding=3),
        ])
        
        self.fusion_conv = nn.Conv1d(output_dim, output_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process micro-doppler features at multiple scales.
        
        Args:
            x: Input features (B, C, T)
            
        Returns:
            Multi-scale processed features (B, output_dim, T)
        """
        # Process at different scales
        scale_outputs = []
        for processor in self.scale_processors:
            scale_output = F.relu(processor(x))
            scale_outputs.append(scale_output)
        
        # Concatenate and fuse
        combined = torch.cat(scale_outputs, dim=1)
        fused = self.fusion_conv(combined)
        
        return fused


# Default configuration
ECHO_NET_CONFIG = {
    "input_dim": 64,
    "out_dim": 256,
    "conv_channels": [64, 128, 256, 512],
    "kernel_sizes": [7, 5, 3, 3],
    "dropout": 0.2
}