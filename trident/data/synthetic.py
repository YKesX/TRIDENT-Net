"""
Synthetic data generation for testing and development.

Author: Yağızhan Keskin
"""

import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional


def generate_rgb_roi(
    batch_size: int = 8,
    channels: int = 3,
    height: int = 224,
    width: int = 224,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic RGB ROI data."""
    # Create base image with some structure
    rgb = torch.randn(batch_size, channels, height, width, device=device)
    
    # Add some spatial structure
    for b in range(batch_size):
        # Add circular or rectangular regions
        center_y, center_x = height // 2, width // 2
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Create a circular region with different intensity
        dist = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        mask = dist < (min(height, width) // 4)
        rgb[b, :, mask] += 0.5 * torch.randn_like(rgb[b, :, mask])
    
    return torch.clamp(rgb, -2, 2)


def generate_rgb_roi_t(
    batch_size: int = 8,
    time_steps: int = 16,
    channels: int = 3,
    height: int = 224,
    width: int = 224,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic temporal RGB ROI data."""
    # Base spatial structure
    base_rgb = generate_rgb_roi(batch_size, channels, height, width, device)
    
    # Expand to temporal dimension with some motion
    rgb_t = base_rgb.unsqueeze(1).repeat(1, time_steps, 1, 1, 1)
    
    # Add temporal variations
    for t in range(time_steps):
        noise_scale = 0.1 * (1 + 0.5 * np.sin(2 * np.pi * t / time_steps))
        rgb_t[:, t] += noise_scale * torch.randn_like(rgb_t[:, t])
    
    return torch.clamp(rgb_t, -2, 2)


def generate_ir_roi_t(
    batch_size: int = 8,
    time_steps: int = 16,
    height: int = 224,
    width: int = 224,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic thermal IR ROI data."""
    # Single channel thermal data
    ir = torch.randn(batch_size, time_steps, 1, height, width, device=device)
    
    # Add hotspots and thermal patterns
    for b in range(batch_size):
        # Random hotspot locations
        n_hotspots = np.random.randint(1, 4)
        for _ in range(n_hotspots):
            hy = np.random.randint(height // 4, 3 * height // 4)
            hx = np.random.randint(width // 4, 3 * width // 4)
            
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing='ij'
            )
            
            # Gaussian hotspot
            dist = torch.sqrt((y_coords - hy)**2 + (x_coords - hx)**2)
            hotspot = torch.exp(-dist**2 / (2 * 20**2))
            
            # Add temporal cooling effect
            for t in range(time_steps):
                cooling_factor = np.exp(-0.1 * t)
                ir[b, t, 0] += 2.0 * cooling_factor * hotspot
    
    return torch.clamp(ir, -2, 2)


def generate_radar_sequence(
    batch_size: int = 8,
    time_steps: int = 128,
    freq_bins: int = 64,
    range_bins: int = 32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic radar micro-Doppler data."""
    # Complex radar returns
    rd_seq = torch.complex(
        torch.randn(batch_size, time_steps, freq_bins, range_bins, device=device),
        torch.randn(batch_size, time_steps, freq_bins, range_bins, device=device)
    )
    
    # Add structured patterns (targets, clutter)
    for b in range(batch_size):
        # Simulated target with Doppler shift
        target_range = np.random.randint(5, range_bins - 5)
        doppler_freq = np.random.randint(-freq_bins//3, freq_bins//3)
        
        for t in range(time_steps):
            # Moving target
            freq_idx = (doppler_freq + freq_bins//2) % freq_bins
            amplitude = 3.0 * np.exp(-0.01 * t)  # Fading
            rd_seq[b, t, freq_idx, target_range] += amplitude * (1 + 0.2j)
    
    # Return magnitude
    return torch.abs(rd_seq)


def generate_pulse_features(
    batch_size: int = 8,
    time_steps: int = 64,
    feature_dim: int = 32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic radar pulse features."""
    # Base features with temporal structure
    pulse_feat = torch.randn(batch_size, time_steps, feature_dim, device=device)
    
    # Add temporal correlations
    for b in range(batch_size):
        # Exponential decay pattern
        decay_rate = np.random.uniform(0.01, 0.1)
        for t in range(1, time_steps):
            correlation = np.exp(-decay_rate * t)
            pulse_feat[b, t] = correlation * pulse_feat[b, 0] + \
                              np.sqrt(1 - correlation**2) * torch.randn(feature_dim, device=device)
    
    return pulse_feat


def generate_radar_tokens(
    batch_size: int = 8,
    seq_length: int = 32,
    token_dim: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic radar transformer tokens."""
    return torch.randn(batch_size, seq_length, token_dim, device=device)


def generate_curve_sequence(
    batch_size: int = 8,
    time_steps: int = 32,
    curve_dim: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate synthetic cooling curve data."""
    # Exponential decay curves with noise
    t_vals = torch.linspace(0, 5, time_steps, device=device)
    curves = torch.zeros(batch_size, time_steps, curve_dim, device=device)
    
    for b in range(batch_size):
        for dim in range(curve_dim):
            # Random decay parameters
            tau = np.random.uniform(0.5, 3.0)
            amplitude = np.random.uniform(0.5, 2.0)
            
            # Exponential decay with noise
            decay_curve = amplitude * torch.exp(-t_vals / tau)
            noise = 0.1 * torch.randn_like(decay_curve)
            curves[b, :, dim] = decay_curve + noise
    
    return curves


def generate_synthetic_batch(
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a complete synthetic batch for testing."""
    if device is None:
        device = torch.device("cpu")
    
    batch = {
        # Visible/EO modality
        "rgb_roi": generate_rgb_roi(batch_size, device=device),
        "rgb_roi_t": generate_rgb_roi_t(batch_size, device=device),
        "rgb_pre": generate_rgb_roi(batch_size, device=device),
        "rgb_post": generate_rgb_roi(batch_size, device=device),
        
        # Thermal/IR modality
        "ir_roi_t": generate_ir_roi_t(batch_size, device=device),
        "curve_seq": generate_curve_sequence(batch_size, device=device),
        
        # Radar modality
        "rd_seq": generate_radar_sequence(batch_size, device=device),
        "pulse_feat": generate_pulse_features(batch_size, device=device),
        "rd_tokens": generate_radar_tokens(batch_size, device=device),
        
        # Labels and metadata
        "y_outcome": torch.randint(0, 2, (batch_size,), device=device),
        "geom": [{"range_m": 1000 + 500 * np.random.randn()} for _ in range(batch_size)],
        "priors": [{"baseline_prob": 0.1 + 0.05 * np.random.randn()} for _ in range(batch_size)],
    }
    
    return batch


def create_synthetic_dataset(
    n_samples: int = 1000,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create synthetic dataset splits.
    
    Args:
        n_samples: Total number of samples
        split_ratios: (train, val, test) ratios
        save_path: Optional path to save dataset
        
    Returns:
        Dict with train/val/test splits
    """
    # Calculate split sizes
    train_size = int(n_samples * split_ratios[0])
    val_size = int(n_samples * split_ratios[1])
    test_size = n_samples - train_size - val_size
    
    splits = {}
    for split_name, size in [("train", train_size), ("val", val_size), ("test", test_size)]:
        if size == 0:
            continue
            
        # Generate samples for this split
        samples = []
        for _ in range(size):
            batch = generate_synthetic_batch(batch_size=1)
            # Remove batch dimension
            sample = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] 
                     for k, v in batch.items()}
            samples.append(sample)
        
        splits[split_name] = samples
    
    if save_path:
        # Save to disk (implement if needed)
        pass
    
    return splits


class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset wrapper for synthetic data."""
    
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate sample on-the-fly with deterministic seed
        torch.manual_seed(idx)
        np.random.seed(idx)
        
        batch = generate_synthetic_batch(batch_size=1)
        sample = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] 
                 for k, v in batch.items()}
        
        return sample