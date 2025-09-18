"""
Accelerate setup for TRIDENT-Net memory-efficient training.

Configures HF Accelerate for automatic GPU/CPU placement with memory limits.
"""

import torch
from accelerate import load_checkpoint_and_dispatch, dispatch_model
from accelerate.utils import get_balanced_memory
from pathlib import Path
from typing import Optional, Dict, Union


def setup_accelerate_model(
    model: torch.nn.Module,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device_map: str = "auto", 
    max_gpu_memory: str = "39GiB",
    max_cpu_memory: str = "30GiB",
    offload_folder: str = "./offload",
    dtype: torch.dtype = torch.float16
) -> torch.nn.Module:
    """
    Setup model with Accelerate device mapping and CPU/disk offload.
    
    Args:
        model: PyTorch model to setup
        checkpoint_path: Path to model checkpoint (optional)
        device_map: Device mapping strategy ("auto", "balanced", etc.)
        max_gpu_memory: Maximum GPU memory to use
        max_cpu_memory: Maximum CPU memory for offload  
        offload_folder: Folder for disk offload
        dtype: Model dtype for mixed precision
        
    Returns:
        Model with device mapping applied
    """
    # Ensure offload folder exists
    Path(offload_folder).mkdir(parents=True, exist_ok=True)
    
    # Define memory limits
    max_memory = {
        0: max_gpu_memory,  # GPU 0
        "cpu": max_cpu_memory
    }
    
    # Set model dtype
    model = model.to(dtype)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        # Load checkpoint with device mapping
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            dtype=dtype
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Dispatch model without loading checkpoint
        model = dispatch_model(
            model,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder
        )
        print("Dispatched model without checkpoint")
    
    # Print device placement info
    if hasattr(model, 'hf_device_map'):
        print("Device mapping:")
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")
    
    return model


def get_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model memory footprint across devices.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with memory usage per device
    """
    memory_map = {}
    
    for name, param in model.named_parameters():
        device = str(param.device)
        param_memory = param.numel() * param.element_size() / 1024**3  # GB
        
        if device not in memory_map:
            memory_map[device] = 0.0
        memory_map[device] += param_memory
    
    return memory_map


def print_model_summary(model: torch.nn.Module) -> None:
    """Print model summary with device placement."""
    print("\n=== Model Summary ===")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
    print(f"Model size: {total_params * 2 / 1024**3:.2f} GB (BF16)")
    
    # Memory footprint per device
    memory_map = get_memory_footprint(model)
    print("\nMemory footprint per device:")
    for device, memory in memory_map.items():
        print(f"  {device}: {memory:.2f} GB")
    
    print("===================\n")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Accelerate model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--max-gpu-mem", type=str, default="39GiB", help="Max GPU memory")
    parser.add_argument("--max-cpu-mem", type=str, default="70GiB", help="Max CPU memory") 
    parser.add_argument("--offload-folder", type=str, default="./offload", help="Offload folder")
    parser.add_argument("--device-map", type=str, default="auto", help="Device mapping strategy")
    
    args = parser.parse_args()
    
    print("Accelerate setup configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Max GPU memory: {args.max_gpu_mem}")
    print(f"  Max CPU memory: {args.max_cpu_mem}")
    print(f"  Offload folder: {args.offload_folder}")
    print(f"  Device map: {args.device_map}")