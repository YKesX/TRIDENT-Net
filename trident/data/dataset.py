"""
Dataset classes for multimodal data loading.

Handles loading and preprocessing of RGB, IR, and radar data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import torch
from torch.utils.data import Dataset
import numpy as np


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal TRIDENT data.
    
    Loads and provides access to RGB, IR, radar, and label data
    in a unified interface.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_rgb: bool = True,
        load_ir: bool = True, 
        load_radar: bool = True,
        sequence_length: int = 8,
        synthetic: bool = False
    ) -> None:
        """
        Initialize multimodal dataset.
        
        Args:
            data_root: Root directory containing data
            split: Data split ('train', 'val', 'test')
            transform: Input transformations
            target_transform: Target transformations
            load_rgb: Whether to load RGB data
            load_ir: Whether to load IR data
            load_radar: Whether to load radar data
            sequence_length: Number of frames for temporal data
            synthetic: Use synthetic data generation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.load_rgb = load_rgb
        self.load_ir = load_ir
        self.load_radar = load_radar
        self.sequence_length = sequence_length
        self.synthetic = synthetic
        
        if synthetic:
            # Use synthetic data
            from .synthetic import SyntheticDataGenerator
            self.generator = SyntheticDataGenerator()
            # Generate a fixed number of samples for consistency
            self.length = 1000 if split == "train" else 200
        else:
            # Load real data index
            self._load_data_index()
    
    def _load_data_index(self) -> None:
        """Load data index from file system."""
        split_file = self.data_root / f"{self.split}.txt"
        
        if split_file.exists():
            with open(split_file, "r") as f:
                self.samples = [line.strip() for line in f.readlines()]
        else:
            # Fallback: search for data files
            self.samples = []
            for ext in [".png", ".jpg", ".npy"]:
                self.samples.extend(
                    [p.stem for p in (self.data_root / self.split).glob(f"*{ext}")]
                )
            self.samples = sorted(list(set(self.samples)))
        
        self.length = len(self.samples)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with multimodal data and labels
        """
        if self.synthetic:
            return self._get_synthetic_sample(idx)
        else:
            return self._get_real_sample(idx)
    
    def _get_synthetic_sample(self, idx: int) -> Dict[str, Any]:
        """Generate synthetic sample."""
        # Set random seed for reproducibility
        np.random.seed(idx)
        torch.manual_seed(idx)
        
        sample = self.generator.generate_sample(
            load_rgb=self.load_rgb,
            load_ir=self.load_ir,
            load_radar=self.load_radar,
            sequence_length=self.sequence_length
        )
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _get_real_sample(self, idx: int) -> Dict[str, Any]:
        """Load real data sample."""
        sample_id = self.samples[idx]
        sample = {}
        
        # Load RGB data
        if self.load_rgb:
            rgb_roi_path = self.data_root / self.split / "rgb" / f"{sample_id}_roi.npy"
            rgb_seq_path = self.data_root / self.split / "rgb" / f"{sample_id}_seq.npy"
            rgb_pre_path = self.data_root / self.split / "rgb" / f"{sample_id}_pre.npy"
            rgb_post_path = self.data_root / self.split / "rgb" / f"{sample_id}_post.npy"
            
            sample.update({
                "rgb_roi": self._load_array(rgb_roi_path),
                "rgb_roi_t": self._load_array(rgb_seq_path),
                "rgb_pre": self._load_array(rgb_pre_path),
                "rgb_post": self._load_array(rgb_post_path),
            })
        
        # Load IR/thermal data
        if self.load_ir:
            ir_roi_path = self.data_root / self.split / "ir" / f"{sample_id}_roi_t.npy"
            sample["ir_roi_t"] = self._load_array(ir_roi_path)
        
        # Load radar data
        if self.load_radar:
            radar_seq_path = self.data_root / self.split / "radar" / f"{sample_id}_seq.npy"
            pulse_feat_path = self.data_root / self.split / "radar" / f"{sample_id}_pulse.npy"
            
            sample.update({
                "rd_seq": self._load_array(radar_seq_path),
                "pulse_feat": self._load_array(pulse_feat_path),
                "rd_tokens": self._load_array(radar_seq_path),  # Placeholder
            })
        
        # Load labels and metadata
        label_path = self.data_root / self.split / "labels" / f"{sample_id}.npy"
        geom_path = self.data_root / self.split / "meta" / f"{sample_id}_geom.npy"
        
        sample.update({
            "y_outcome": self._load_array(label_path, default=0),
            "geom": self._load_dict(geom_path, default={}),
            "priors": {"detection_threshold": 0.5, "confidence_min": 0.8},
        })
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_array(
        self, 
        path: Path, 
        default: Optional[Union[int, float, np.ndarray]] = None
    ) -> torch.Tensor:
        """
        Load numpy array and convert to tensor.
        
        Args:
            path: File path
            default: Default value if file doesn't exist
            
        Returns:
            PyTorch tensor
        """
        if path.exists():
            data = np.load(path)
            return torch.from_numpy(data).float()
        elif default is not None:
            if isinstance(default, (int, float)):
                return torch.tensor(default).float()
            else:
                return torch.from_numpy(default).float()
        else:
            raise FileNotFoundError(f"Data file not found: {path}")
    
    def _load_dict(
        self, 
        path: Path, 
        default: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Load dictionary from file.
        
        Args:
            path: File path
            default: Default value if file doesn't exist
            
        Returns:
            Dictionary
        """
        if path.exists():
            data = np.load(path, allow_pickle=True).item()
            return data if isinstance(data, dict) else {}
        else:
            return default or {}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for multimodal batches.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    if not batch:
        return {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        if key in ["geom", "priors"]:
            # Keep as list for metadata
            batched[key] = values
        elif key in ["y_outcome"] and all(isinstance(v, (int, float)) for v in values):
            # Convert scalar labels to tensor
            batched[key] = torch.tensor(values).float()
        elif all(isinstance(v, torch.Tensor) for v in values):
            # Stack tensors
            try:
                batched[key] = torch.stack(values)
            except RuntimeError:
                # Handle variable-sized tensors with padding
                max_shape = tuple(max(v.shape[i] for v in values) 
                                for i in range(values[0].dim()))
                padded_values = []
                for v in values:
                    pad_amounts = []
                    for i in range(v.dim()):
                        pad_amounts.extend([0, max_shape[i] - v.shape[i]])
                    padded_values.append(torch.nn.functional.pad(v, pad_amounts))
                batched[key] = torch.stack(padded_values)
        else:
            # Keep as list for other types
            batched[key] = values
    
    return batched


class DataTransforms:
    """Common data transformations for TRIDENT dataset."""
    
    @staticmethod
    def normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
        """Normalize RGB values to [0,1]."""
        return rgb / 255.0 if rgb.max() > 1.0 else rgb
    
    @staticmethod
    def normalize_thermal(thermal: torch.Tensor) -> torch.Tensor:
        """Normalize thermal data using percentile scaling."""
        p1, p99 = torch.quantile(thermal, torch.tensor([0.01, 0.99]))
        return torch.clamp((thermal - p1) / (p99 - p1), 0, 1)
    
    @staticmethod
    def augment_spatial(
        image: torch.Tensor, 
        prob: float = 0.5
    ) -> torch.Tensor:
        """Apply spatial augmentations."""
        if torch.rand(1) < prob and image.dim() >= 3:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                image = torch.flip(image, [-1])
            # Random rotation (90 degree multiples)
            if torch.rand(1) < 0.5:
                k = torch.randint(1, 4, (1,)).item()
                image = torch.rot90(image, k, [-2, -1])
        return image