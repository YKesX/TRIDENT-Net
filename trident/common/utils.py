"""
Utility functions for the TRIDENT system.

Common helper functions used across multiple modules.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
import numpy as np


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Set up logging configuration for TRIDENT system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        force_cpu: Force CPU usage even if CUDA is available
        
    Returns:
        PyTorch device object
    """
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    **kwargs: Any
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        path: Checkpoint file path
        **kwargs: Additional state to save
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **kwargs
    }
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        model: PyTorch model to load state into
        path: Checkpoint file path
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint metadata (epoch, loss, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {k: v for k, v in checkpoint.items() 
            if k not in ["model_state_dict", "optimizer_state_dict"]}


def ensure_tensor(
    data: Union[np.ndarray, torch.Tensor, list, float], 
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Convert input data to PyTorch tensor.
    
    Args:
        data: Input data to convert
        device: Target device
        dtype: Target dtype
        
    Returns:
        PyTorch tensor
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    
    if dtype is not None:
        data = data.to(dtype)
    if device is not None:
        data = data.to(device)
        
    return data


def apply_gradient_clipping(
    model: torch.nn.Module, 
    max_norm: float = 1.0
) -> float:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


class MovingAverage:
    """Exponential moving average tracker."""
    
    def __init__(self, decay: float = 0.99) -> None:
        """
        Initialize moving average.
        
        Args:
            decay: Decay factor for exponential averaging
        """
        self.decay = decay
        self.value: Optional[float] = None
        
    def update(self, new_value: float) -> float:
        """
        Update moving average with new value.
        
        Args:
            new_value: New value to incorporate
            
        Returns:
            Updated moving average
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value
    
    def reset(self) -> None:
        """Reset moving average."""
        self.value = None