"""
Synthetic data generation for testing and development.

Author: Yağızhan Keskin
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import torch
import torch.utils.data

from ..common.types import EventToken


class SyntheticVideoJsonl:
    """
    Generate synthetic JSONL dataset for VideoJsonlDataset testing.
    
    Creates realistic video metadata with timing information matching
    the shoot→hit→kill hierarchy and video clip structure.
    """
    
    def __init__(
        self,
        count: int = 64,
        clip_seconds: float = 3.6,
        ensure_hit_rate: float = 0.5,
        ensure_kill_given_hit: float = 0.6,
        fps: int = 24,
        seed: int = 12345
    ):
        """
        Initialize synthetic JSONL generator.
        
        Args:
            count: Number of samples to generate
            clip_seconds: Duration of video clips in seconds
            ensure_hit_rate: Probability of hit given shoot
            ensure_kill_given_hit: Probability of kill given hit
            fps: Frames per second for timing calculations
            seed: Random seed for reproducible generation
        """
        self.count = count
        self.clip_seconds = clip_seconds
        self.ensure_hit_rate = ensure_hit_rate
        self.ensure_kill_given_hit = ensure_kill_given_hit
        self.fps = fps
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def generate_sample(self, idx: int) -> Dict[str, Any]:
        """
        Generate single synthetic sample with realistic timing.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with JSONL-compatible structure
        """
        # Use index for per-sample reproducibility
        np.random.seed(self.seed + idx)
        
        # Generate timing hierarchy: shoot → hit → kill
        clip_ms = int(self.clip_seconds * 1000)
        
        # Shoot always happens (reference point)
        shoot_ms = np.random.randint(800, clip_ms - 800)  # Not too close to edges
        
        # Hit happens with probability ensure_hit_rate
        hit_ms = None
        kill_ms = None
        
        if np.random.random() < self.ensure_hit_rate:
            # Hit occurs 100-800ms after shoot
            hit_delay = np.random.randint(100, 800)
            hit_ms = shoot_ms + hit_delay
            
            # Kill happens with probability ensure_kill_given_hit
            if np.random.random() < self.ensure_kill_given_hit:
                # Kill occurs 200-1000ms after hit
                kill_delay = np.random.randint(200, 1000)
                kill_ms = hit_ms + kill_delay
                
                # Ensure kill doesn't exceed clip duration
                if kill_ms >= clip_ms:
                    kill_ms = None
        
        # Generate synthetic video paths
        video_id = f"synthetic_{idx:06d}"
        rgb_path = f"videos/rgb/{video_id}.mp4"
        ir_path = f"videos/ir/{video_id}.mp4" if np.random.random() < 0.8 else None  # 80% have IR
        
        # Generate synthetic kinematics data (3x9 for pre/fire/post)
        kinematics = np.random.randn(3, 9).astype(float).tolist()
        
        # Optional class ID (8 classes)
        class_id = np.random.randint(0, 8) if np.random.random() < 0.7 else None
        class_conf = np.random.uniform(0.5, 1.0) if class_id is not None else None
        
        # Generate synthetic bounding box (optional)
        has_bbox = np.random.random() < 0.6
        bbox = None
        if has_bbox:
            x = np.random.uniform(100, 1000)
            y = np.random.uniform(100, 600)
            w = np.random.uniform(50, 200)
            h = np.random.uniform(50, 150)
            bbox = [x, y, w, h]
        
        # Create sample structure matching tasks.yml field mapping
        sample = {
            "video": {
                "path": rgb_path,
                "rgb_path": rgb_path,
            },
            "radar": {
                "kinematics": kinematics
            },
            "prompt": f"Analyze target engagement sequence for sample {idx}",
            
            # Timing data
            "shoot_ms": shoot_ms,
            "hit_ms": hit_ms,
            "kill_ms": kill_ms,
            
            # Target data
            "target": {}
        }
        
        # Add IR path if available
        if ir_path:
            sample["video"]["ir_path"] = ir_path
        
        # Add class information if available
        if class_id is not None:
            sample["target"]["class_id"] = class_id
        if class_conf is not None:
            sample["target"]["class_conf"] = class_conf
        
        # Add bounding box if available
        if bbox is not None:
            sample["target"]["bbox"] = bbox
        
        return sample
    
    def generate_jsonl_file(self, output_path: str) -> str:
        """
        Generate complete JSONL file with all samples.
        
        Args:
            output_path: Path to save JSONL file
            
        Returns:
            Path to created JSONL file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for idx in range(self.count):
                sample = self.generate_sample(idx)
                f.write(json.dumps(sample) + '\n')
        
        return str(output_path)
    
    def create_synthetic_videos(self, video_root: str) -> None:
        """
        Create synthetic video files (dummy) for testing.
        
        Args:
            video_root: Root directory for video files
        """
        video_root = Path(video_root)
        rgb_dir = video_root / "videos" / "rgb"
        ir_dir = video_root / "videos" / "ir"
        
        rgb_dir.mkdir(parents=True, exist_ok=True)
        ir_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy video files (just empty files for testing)
        for idx in range(self.count):
            video_id = f"synthetic_{idx:06d}"
            
            # RGB video (always present)
            rgb_path = rgb_dir / f"{video_id}.mp4"
            rgb_path.write_bytes(b'dummy_rgb_video_data')
            
            # IR video (80% present)
            if np.random.random() < 0.8:
                ir_path = ir_dir / f"{video_id}.mp4"
                ir_path.write_bytes(b'dummy_ir_video_data')
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about generated dataset."""
        hit_count = 0
        kill_count = 0
        ir_count = 0
        
        for idx in range(self.count):
            sample = self.generate_sample(idx)
            
            if sample['hit_ms'] is not None:
                hit_count += 1
                
            if sample['kill_ms'] is not None:
                kill_count += 1
                
            if sample['video'].get('ir_path') is not None:
                ir_count += 1
        
        return {
            'total_samples': self.count,
            'hit_rate': hit_count / self.count,
            'kill_rate': kill_count / self.count,
            'kill_given_hit_rate': kill_count / max(hit_count, 1),
            'ir_availability_rate': ir_count / self.count
        }


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


def generate_synthetic_batch_variable_t(
    batch_size: int = 2,
    T: int = 36,  # Variable temporal dimension
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic batch for variable-T video testing.
    
    Expected shapes matching tasks.yml:
    - rgb: B x 3 x T x 704 x 1248
    - ir:  B x 1 x T x 704 x 1248  
    - kin: B x 3 x 9 (pre, fire, post)
    - class_id: B
    - labels: {hit: B x 1, kill: B x 1}
    """
    if device is None:
        device = torch.device("cpu")
    
    # Generate video clips with synthetic motion
    rgb = torch.randn(batch_size, 3, T, 704, 1248, device=device)
    ir = torch.randn(batch_size, 1, T, 704, 1248, device=device)
    
    # Add some temporal structure
    for b in range(batch_size):
        for t in range(T):
            # Add moving objects
            x_pos = int(200 + 100 * np.sin(2 * np.pi * t / T))
            y_pos = int(300 + 50 * np.cos(2 * np.pi * t / T))
            
            # RGB object
            if x_pos < 1200 and y_pos < 650:
                rgb[b, :, t, y_pos:y_pos+54, x_pos:x_pos+48] += 0.5
            
            # IR hotspot (delayed relative to RGB)
            ir_delay = max(0, t - 5)
            ir_x = int(200 + 100 * np.sin(2 * np.pi * ir_delay / T))
            ir_y = int(300 + 50 * np.cos(2 * np.pi * ir_delay / T))
            
            if ir_x < 1200 and ir_y < 650:
                ir[b, 0, t, ir_y:ir_y+54, ir_x:ir_x+48] += 1.0
    
    # Kinematics for (pre, fire, post) windows  
    kin = torch.randn(batch_size, 3, 9, device=device)
    
    # Optional class IDs
    class_id = torch.randint(0, 8, (batch_size,), device=device)
    
    # Binary labels with hierarchy: kill ⊆ hit
    hit_labels = torch.randint(0, 2, (batch_size, 1), device=device).float()
    kill_labels = torch.zeros_like(hit_labels)
    
    # Ensure hierarchy: if hit=1, kill can be 0 or 1; if hit=0, kill must be 0
    for b in range(batch_size):
        if hit_labels[b, 0] == 1.0:
            kill_labels[b, 0] = float(torch.rand(1).item() < 0.6)  # 60% kill rate given hit
    
    # Times metadata (milliseconds)
    times_ms = []
    for b in range(batch_size):
        shoot_ms = 1500  # Fixed shoot time
        hit_ms = 2200 if hit_labels[b, 0] == 1.0 else None
        kill_ms = 3000 if kill_labels[b, 0] == 1.0 else None
        
        times_ms.append({
            'shoot_ms': shoot_ms,
            'hit_ms': hit_ms, 
            'kill_ms': kill_ms
        })
    
    batch = {
        'rgb': rgb,
        'ir': ir,
        'kin': kin,
        'class_id': class_id,
        'labels': {
            'hit': hit_labels,
            'kill': kill_labels
        },
        'meta': [{'times_ms': times_ms[b], 'sample_idx': b} for b in range(batch_size)]
    }
    
    return batch


def generate_synthetic_batch(
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate synthetic batch following tasks.yml specification.
    
    Expected shapes from tasks.yml:
    - rgb_seq: B x 3 x 3 x 768 x 1120 (batch, time, channels, H, W)
    - ir_seq:  B x 3 x 1 x 768 x 1120  
    - k_seq:   B x 3 x 9 (batch, time, features)
    - class_id: B
    - labels: {hit: B x 1, kill: B x 1}
    """
    if device is None:
        device = torch.device("cpu")
    
    batch = {
        # RGB sequence: B x 3 x 3 x 768 x 1120 (3 frames, 3 channels each)
        "rgb_seq": torch.rand(batch_size, 3, 3, 768, 1120, device=device),
        
        # IR sequence: B x 3 x 1 x 768 x 1120 (3 frames, 1 channel each)  
        "ir_seq": torch.rand(batch_size, 3, 1, 768, 1120, device=device),
        
        # Kinematics sequence: B x 3 x 9 (3 timesteps, 9 features)
        # Features: ["x","y","z","vx","vy","vz","range","bearing","elevation"]
        "k_seq": torch.randn(batch_size, 3, 9, device=device),
        
        # Optional class ID
        "class_id": torch.randint(0, 8, (batch_size,), device=device),
        
        # Outcome labels
        "y_outcome": {
            "hit": torch.rand(batch_size, 1, device=device),   # [0,1] probabilities
            "kill": torch.rand(batch_size, 1, device=device),  # [0,1] probabilities
        },
        
        # Metadata
        "meta": [{"sample_id": i} for i in range(batch_size)],
        
        # Geometry and priors for guard module
        "geom": [{"bearing": 0.5 * np.random.randn(), "elevation": 0.2 * np.random.randn()} for _ in range(batch_size)],
        "priors": [{"baseline_prob": 0.1 + 0.05 * np.random.randn()} for _ in range(batch_size)],
    }
    
    return batch


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for testing with variable T support.
    
    Generates data on-demand to match tasks.yml specification.
    """
    
    def __init__(
        self, 
        n_samples: int = 100,
        variable_t: bool = True,
        min_t: int = 20,
        max_t: int = 50
    ):
        """
        Args:
            n_samples: Number of samples
            variable_t: Whether to use variable temporal dimension
            min_t: Minimum T frames if variable_t=True
            max_t: Maximum T frames if variable_t=True
        """
        self.n_samples = n_samples
        self.variable_t = variable_t
        self.min_t = min_t
        self.max_t = max_t
        
    def __len__(self) -> int:
        return self.n_samples
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a single sample."""
        # Use the index as a seed for reproducibility
        torch.manual_seed(idx + 42)
        np.random.seed(idx + 42)
        
        if self.variable_t:
            # Variable temporal dimension
            T = np.random.randint(self.min_t, self.max_t + 1)
            batch = generate_synthetic_batch_variable_t(batch_size=1, T=T)
        else:
            # Fixed temporal dimension 
            batch = generate_synthetic_batch(batch_size=1)
        
        # Remove batch dimension to get individual sample
        sample = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.squeeze(0)  # Remove batch dimension
            elif isinstance(value, dict):
                # Handle nested dict (like labels)
                sample[key] = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                             for k, v in value.items()}
            elif isinstance(value, list):
                sample[key] = value[0]  # Take first element
            else:
                sample[key] = value
                
        return sample


def create_synthetic_test_data(
    temp_dir: Optional[str] = None,
    count: int = 32
) -> Tuple[str, str]:
    """
    Create synthetic JSONL dataset for testing.
    
    Args:
        temp_dir: Optional temporary directory (uses system temp if None)
        count: Number of samples to generate
        
    Returns:
        tuple: (jsonl_path, video_root)
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    temp_dir = Path(temp_dir)
    
    # Generate synthetic JSONL dataset
    generator = SyntheticVideoJsonl(
        count=count,
        clip_seconds=3.6,
        ensure_hit_rate=0.5,
        ensure_kill_given_hit=0.6
    )
    
    # Create JSONL file
    jsonl_path = generator.generate_jsonl_file(temp_dir / "synthetic_data.jsonl")
    
    # Create dummy video files
    video_root = str(temp_dir)
    generator.create_synthetic_videos(video_root)
    
    return jsonl_path, video_root