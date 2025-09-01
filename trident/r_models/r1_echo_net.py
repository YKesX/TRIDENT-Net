"""
TRIDENT-R1: Echo Network (1D CNN for micro-Doppler)

Author: Yağızhan Keskin
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken, FeatureVec


class Conv1DBlock(nn.Module):
    """1D Convolutional block with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)


class MicroDopplerAnalyzer(nn.Module):
    """Specialized analyzer for micro-Doppler signatures."""
    
    def __init__(self, freq_bins: int = 64):
        super().__init__()
        self.freq_bins = freq_bins
        
        # Frequency domain analysis
        self.freq_conv = nn.Conv1d(freq_bins, 32, kernel_size=3, padding=1)
        self.freq_pool = nn.AdaptiveAvgPool1d(1)
        
        # Doppler shift detection
        self.doppler_detector = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # RCS analysis
        self.rcs_analyzer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, freq_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze micro-Doppler characteristics.
        
        Args:
            freq_features: Frequency features of shape (B, F, T)
            
        Returns:
            tuple: (doppler_shift, rcs_estimate)
        """
        # Process frequency features
        freq_processed = F.relu(self.freq_conv(freq_features))  # (B, 32, T)
        freq_pooled = self.freq_pool(freq_processed).squeeze(-1)  # (B, 32)
        
        # Detect Doppler characteristics
        doppler_shift = self.doppler_detector(freq_pooled)
        rcs_estimate = self.rcs_analyzer(freq_pooled)
        
        return doppler_shift, rcs_estimate


class EchoNet(BranchModule):
    """
    Echo Network for processing radar micro-Doppler sequences.
    
    Uses 1D CNN to process time-frequency micro-Doppler data
    and extract target characteristics and events.
    """
    
    def __init__(
        self,
        conv_blocks: int = 4,
        out_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(out_dim)
        
        self.conv_blocks = conv_blocks
        self.base_channels = base_channels
        
        # Input dimensions will be inferred from first forward pass
        self.input_dim = None
        self.freq_bins = None
        self.range_bins = None
        
        # Create 1D CNN blocks (will be initialized in first forward)
        self.conv_layers = None
        self.micro_doppler_analyzer = None
        self.feature_head = None
        self.event_detectors = None
        
        self.dropout = dropout
        self._initialized = False
    
    def _initialize_layers(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize layers based on input shape."""
        B, T, F, R = input_shape
        self.freq_bins = F
        self.range_bins = R
        
        # Flatten spatial dimensions for 1D processing
        self.input_dim = F * R
        
        # Create 1D CNN layers
        layers = []
        channels = [self.input_dim] + [self.base_channels * (2**i) for i in range(self.conv_blocks)]
        
        for i in range(self.conv_blocks):
            layers.append(Conv1DBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                stride=2 if i % 2 == 1 else 1,  # Downsample every other layer
                dropout=self.dropout,
            ))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Micro-Doppler analyzer
        self.micro_doppler_analyzer = MicroDopplerAnalyzer(freq_bins=F)
        
        # Feature head
        final_channels = channels[-1]
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_channels, self.out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
        )
        
        # Event detection heads
        self.event_detectors = nn.ModuleDict({
            'rcs_drop': nn.Sequential(
                nn.Linear(self.out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ),
            'decorrelation': nn.Sequential(
                nn.Linear(self.out_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ),
            'velocity_change': nn.Sequential(
                nn.Linear(self.out_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ),
        })
        
        self._initialized = True
    
    def forward(self, rd_seq: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for radar echo processing.
        
        Args:
            rd_seq: Radar sequence of shape (B, T, F, R) where:
                    T = time steps, F = frequency bins, R = range bins
                    
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        if not self._initialized:
            self._initialize_layers(rd_seq.shape)
        
        B, T, F, R = rd_seq.shape
        
        # Reshape for 1D processing: (B, F*R, T)
        rd_flat = rd_seq.view(B, T, F * R).permute(0, 2, 1)
        
        # Apply 1D CNN layers
        conv_features = self.conv_layers(rd_flat)  # (B, channels, T')
        
        # Extract global features
        global_features = self.feature_head(conv_features)  # (B, out_dim)
        
        # Analyze micro-Doppler characteristics on original frequency data
        freq_data = rd_seq.mean(dim=-1)  # Average over range: (B, T, F)
        freq_data = freq_data.permute(0, 2, 1)  # (B, F, T)
        doppler_shift, rcs_estimate = self.micro_doppler_analyzer(freq_data)
        
        # Event detection
        events_output = {}
        for event_type, detector in self.event_detectors.items():
            events_output[event_type] = detector(global_features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=global_features)
        
        # Create EventTokens
        events = []
        for b in range(B):
            # RCS drop event
            rcs_drop_db = events_output['rcs_drop'][b, 0].item()
            if rcs_drop_db > 3.0:  # Significant RCS drop
                events.append(EventToken(
                    type="rcs_drop_db",
                    value=rcs_drop_db,
                    t_start=0.0,
                    t_end=T / 1000.0,  # Assume 1kHz sampling
                    quality=min(rcs_drop_db / 20.0, 1.0),  # Normalize by max expected drop
                    meta={
                        "rcs_estimate": rcs_estimate[b, 0].item(),
                        "doppler_shift": doppler_shift[b, 0].item(),
                        "frequency_bins": F,
                        "range_bins": R,
                    }
                ))
            
            # Decorrelation event
            decorr_prob = events_output['decorrelation'][b, 0].item()
            if decorr_prob > 0.6:
                events.append(EventToken(
                    type="decorrelation",
                    value=decorr_prob,
                    t_start=0.0,
                    t_end=T / 1000.0,
                    quality=decorr_prob,
                    meta={
                        "correlation_loss": 1.0 - decorr_prob,
                        "signal_coherence": "degraded",
                    }
                ))
            
            # Velocity change event
            vel_change = events_output['velocity_change'][b, 0].item()
            if abs(vel_change) > 2.0:  # m/s
                events.append(EventToken(
                    type="velocity_change_mps",
                    value=vel_change,
                    t_start=0.0,
                    t_end=T / 1000.0,
                    quality=min(abs(vel_change) / 10.0, 1.0),
                    meta={
                        "velocity_direction": "acceleration" if vel_change > 0 else "deceleration",
                        "magnitude_mps": abs(vel_change),
                        "doppler_shift_hz": doppler_shift[b, 0].item(),
                    }
                ))
        
        # Store intermediate outputs
        self._last_conv_features = conv_features
        self._last_doppler_analysis = (doppler_shift, rcs_estimate)
        self._last_events_raw = events_output
        
        return feature_vec, events
    
    def get_conv_features(self) -> torch.Tensor:
        """Get 1D CNN feature maps."""
        if hasattr(self, '_last_conv_features'):
            return self._last_conv_features
        else:
            raise RuntimeError("No conv features available. Run forward() first.")
    
    def get_doppler_analysis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Doppler shift and RCS analysis results."""
        if hasattr(self, '_last_doppler_analysis'):
            return self._last_doppler_analysis
        else:
            raise RuntimeError("No Doppler analysis available. Run forward() first.")
    
    def get_event_scores(self) -> dict:
        """Get raw event detection scores."""
        if hasattr(self, '_last_events_raw'):
            return self._last_events_raw
        else:
            raise RuntimeError("No event scores available. Run forward() first.")


def create_echo_net(config: dict) -> EchoNet:
    """Factory function to create EchoNet from config."""
    return EchoNet(
        conv_blocks=config.get("conv_blocks", 4),
        out_dim=config.get("out_dim", 256),
        base_channels=config.get("base_channels", 32),
        dropout=config.get("dropout", 0.1),
    )