"""
Dataset and data loading utilities for TRIDENT-Net.

Author: Yağızhan Keskin
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal sensor data.
    
    Expected data structure:
    data_root/
        rgb_roi/
        rgb_roi_t/
        rgb_pre/
        rgb_post/
        ir_roi_t/
        rd_seq/
        pulse_feat/
        rd_tokens/
        labels.npy
        metadata.npy
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        transform: Optional[callable] = None,
        load_all_modalities: bool = True,
    ) -> None:
        """
        Args:
            data_root: Root directory containing data
            split: Data split ('train', 'val', 'test')
            transform: Optional data transformation function
            load_all_modalities: Whether to load all modalities or subset
        """
        self.data_root = Path(data_root) / split
        self.transform = transform
        self.load_all_modalities = load_all_modalities
        
        # Load metadata and labels
        self.labels = np.load(self.data_root / "labels.npy")
        self.metadata = np.load(self.data_root / "metadata.npy", allow_pickle=True)
        
        self.length = len(self.labels)
        
        # Check which modalities are available
        self.available_modalities = self._check_available_modalities()
    
    def _check_available_modalities(self) -> Dict[str, bool]:
        """Check which modality directories exist."""
        modalities = {
            "rgb_roi": "rgb_roi",
            "rgb_roi_t": "rgb_roi_t", 
            "rgb_pre": "rgb_pre",
            "rgb_post": "rgb_post",
            "ir_roi_t": "ir_roi_t",
            "rd_seq": "rd_seq",
            "pulse_feat": "pulse_feat",
            "rd_tokens": "rd_tokens",
        }
        
        available = {}
        for key, dirname in modalities.items():
            available[key] = (self.data_root / dirname).exists()
        
        return available
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample at given index."""
        sample = {}
        
        # Load labels and metadata
        sample["y_outcome"] = self.labels[idx]
        sample["meta"] = self.metadata[idx] if len(self.metadata) > idx else {}
        
        # Load available modalities
        if self.available_modalities.get("rgb_roi", False):
            sample["rgb_roi"] = self._load_tensor(f"rgb_roi/{idx:06d}.npy")
            
        if self.available_modalities.get("rgb_roi_t", False):
            sample["rgb_roi_t"] = self._load_tensor(f"rgb_roi_t/{idx:06d}.npy")
            
        if self.available_modalities.get("rgb_pre", False):
            sample["rgb_pre"] = self._load_tensor(f"rgb_pre/{idx:06d}.npy")
            
        if self.available_modalities.get("rgb_post", False):
            sample["rgb_post"] = self._load_tensor(f"rgb_post/{idx:06d}.npy")
            
        if self.available_modalities.get("ir_roi_t", False):
            sample["ir_roi_t"] = self._load_tensor(f"ir_roi_t/{idx:06d}.npy")
            
        if self.available_modalities.get("rd_seq", False):
            sample["rd_seq"] = self._load_tensor(f"rd_seq/{idx:06d}.npy")
            
        if self.available_modalities.get("pulse_feat", False):
            sample["pulse_feat"] = self._load_tensor(f"pulse_feat/{idx:06d}.npy")
            
        if self.available_modalities.get("rd_tokens", False):
            sample["rd_tokens"] = self._load_tensor(f"rd_tokens/{idx:06d}.npy")
        
        # Add geometric and prior information from metadata
        meta = sample["meta"]
        sample["geom"] = meta.get("geometry", {})
        sample["priors"] = meta.get("priors", {})
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_tensor(self, rel_path: str) -> torch.Tensor:
        """Load numpy array and convert to tensor."""
        try:
            data = np.load(self.data_root / rel_path)
            return torch.from_numpy(data).float()
        except FileNotFoundError:
            # Return dummy tensor for missing files
            return torch.zeros(1)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a sample without loading data."""
        meta = self.metadata[idx] if len(self.metadata) > idx else {}
        return {
            "index": idx,
            "label": self.labels[idx],
            "metadata": meta,
            "available_modalities": [k for k, v in self.available_modalities.items() if v],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for multimodal batches.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors
    """
    if not batch:
        return {}
    
    # Get all keys from first sample
    keys = set(batch[0].keys())
    
    # Separate tensor and non-tensor keys
    tensor_keys = []
    other_keys = []
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            tensor_keys.append(key)
        else:
            other_keys.append(key)
    
    result = {}
    
    # Stack tensor data
    for key in tensor_keys:
        tensors = [sample[key] for sample in batch if key in sample]
        if tensors:
            try:
                result[key] = torch.stack(tensors, dim=0)
            except RuntimeError:
                # Handle variable size tensors by padding
                result[key] = _pad_and_stack(tensors)
    
    # Collect non-tensor data
    for key in other_keys:
        result[key] = [sample.get(key, None) for sample in batch]
    
    return result


def _pad_and_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad tensors to same size and stack."""
    if not tensors:
        return torch.empty(0)
    
    # Find maximum dimensions
    max_dims = []
    for dim in range(tensors[0].dim()):
        max_size = max(t.shape[dim] for t in tensors)
        max_dims.append(max_size)
    
    # Pad tensors
    padded = []
    for tensor in tensors:
        pad_widths = []
        for dim in range(tensor.dim()):
            pad_width = max_dims[dim] - tensor.shape[dim]
            pad_widths.extend([0, pad_width])
        
        # Reverse padding for F.pad (needs last-dim-first order)
        pad_widths = pad_widths[::-1]
        padded_tensor = torch.nn.functional.pad(tensor, pad_widths)
        padded.append(padded_tensor)
    
    return torch.stack(padded, dim=0)


class MultimodalSubset(Dataset):
    """Subset of multimodal dataset with specific modalities."""
    
    def __init__(
        self,
        dataset: MultimodalDataset,
        modalities: List[str],
        indices: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            dataset: Parent multimodal dataset
            modalities: List of modalities to include
            indices: Optional subset of indices to use
        """
        self.dataset = dataset
        self.modalities = set(modalities)
        self.indices = indices if indices is not None else list(range(len(dataset)))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get filtered sample."""
        actual_idx = self.indices[idx]
        full_sample = self.dataset[actual_idx]
        
        # Filter to requested modalities + always include labels and metadata
        keep_keys = self.modalities | {"y_outcome", "meta", "geom", "priors"}
        filtered_sample = {k: v for k, v in full_sample.items() if k in keep_keys}
        
        return filtered_sample


def create_data_loaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[callable] = None,
    val_transform: Optional[callable] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test data loaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else val_transform
        try:
            datasets[split] = MultimodalDataset(
                data_root=data_root,
                split=split,
                transform=transform,
            )
        except FileNotFoundError:
            datasets[split] = None
    
    loaders = []
    for split in ["train", "val", "test"]:
        if datasets[split] is not None:
            shuffle = (split == "train")
            loader = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
            loaders.append(loader)
        else:
            loaders.append(None)
    
    return tuple(loaders)