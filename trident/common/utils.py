"""
Utility functions for TRIDENT-Net.

Author: Yağızhan Keskin
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module.""" 
    for param in module.parameters():
        param.requires_grad = True


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch tensors to specified device."""
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = move_to_device(value, device)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            result[key] = [v.to(device) for v in value]
        else:
            result[key] = value
    return result


class AverageMeter:
    """Tracks running average of values."""
    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def start(self) -> None:
        self.start_time = time.time()
    
    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def __enter__(self) -> "Timer":
        self.start()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.stop()


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def tensor_info(tensor: torch.Tensor, name: str = "tensor") -> str:
    """Get tensor information string."""
    return (
        f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
        f"device={tensor.device}, min={tensor.min():.4f}, "
        f"max={tensor.max():.4f}, mean={tensor.mean():.4f}"
    )