"""
Smoke test for memory-efficient TRIDENT-Net training.

Tests both DeepSpeed and Accelerate variants with synthetic data.
Validates that VRAM usage stays under 39 GiB.
"""

import os
import sys
import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trident.runtime.memory_efficient_trainer import MemoryEfficientTrainer
from trident.runtime.config import ConfigLoader
from trident.data.synthetic import generate_synthetic_batch
from trident.runtime.cli import _build_models

# Ensure we're using CPU for this test if no GPU available
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def create_test_config() -> Dict[str, Any]:
    """Create minimal test configuration."""
    return {
        'environment': {
            'seed': 12345,
            'device': 'auto'
        },
        'data': {
            'loader': {
                'batch_size': 1,  # Small batch for testing
                'num_workers': 0,
                'pin_memory': False
            },
            'synthetic': {
                'enabled': True,
                'count': 4,
                'clip_seconds': 3.6
            }
        },
        'training': {
            'optimizer': {
                'type': 'adamw',
                'lr': 2e-4,
                'weight_decay': 0.01
            },
            'amp': True,
            'grad_clip_norm': 1.0,
            'epochs': {
                'train_fusion': 1
            }
        },
        'components': {
            'fusion_guard': {
                'f2': {
                    'd_model': 256,  # Smaller for testing
                    'n_heads': 4,
                    'n_layers': 2
                }
            }
        }
    }


def create_synthetic_data(batch_size: int = 2) -> Dict[str, torch.Tensor]:
    """Create synthetic training batch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create smaller tensors for testing
    data = {
        'rgb': torch.randn(batch_size, 3, 8, 256, 256, device=device),  # Smaller than full resolution
        'ir': torch.randn(batch_size, 1, 8, 256, 256, device=device),
        'kin': torch.randn(batch_size, 3, 9, device=device),
        'class_id': torch.randint(0, 10, (batch_size,), device=device),
        'labels': {
            'hit': torch.rand(batch_size, 1, device=device),
            'kill': torch.rand(batch_size, 1, device=device)
        }
    }
    return data


def get_memory_usage() -> Tuple[float, float]:
    """Get current and peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return current, peak
    return 0.0, 0.0


def reset_memory_stats():
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def test_deepspeed_variant() -> Dict[str, Any]:
    """Test DeepSpeed ZeRO-2 offload variant."""
    print("\n=== Testing DeepSpeed ZeRO-2 Offload Variant ===")
    
    reset_memory_stats()
    
    config = create_test_config()
    config_loader = ConfigLoader(config)
    
    # Create trainer with DeepSpeed
    trainer = MemoryEfficientTrainer(
        config_loader=config_loader,
        use_bf16=True,
        use_checkpointing=True,
        use_8bit_optimizer=True,
        optimizer_type="adamw8bit",
        grad_accum_steps=4,
        use_deepspeed=True,
        deepspeed_config={
            "train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},
                "overlap_comm": True,
                "contiguous_gradients": True
            }
        }
    )
    
    # Create simple model for testing
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d = torch.nn.Conv3d(3, 64, kernel_size=3, padding=1)
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, batch_first=True
                ), num_layers=2
            )
            self.fc = torch.nn.Linear(256, 2)
            
        def forward(self, rgb, **kwargs):
            # Simple forward for testing
            B = rgb.shape[0]
            x = self.conv3d(rgb[:, :, :3])  # Take first 3 frames
            x = x.mean([2, 3, 4])  # Global average pool
            x = x.unsqueeze(1).expand(-1, 10, -1)  # Fake sequence
            x = self.transformer(x)
            x = x.mean(1)  # Average over sequence
            out = self.fc(x)
            return {'p_hit': torch.sigmoid(out[:, 0:1]), 'p_kill': torch.sigmoid(out[:, 1:2])}
    
    model = SimpleTestModel()
    
    try:
        # Prepare model
        model, optimizer = trainer.prepare_model_for_training(model, "test_model")
        
        # Create synthetic data
        batch = create_synthetic_data(batch_size=2)
        
        # Training step
        start_mem = get_memory_usage()
        metrics = trainer.training_step(model, batch, optimizer, step=0)
        end_mem = get_memory_usage()
        
        results = {
            'variant': 'deepspeed',
            'success': True,
            'memory_start_gb': start_mem[0],
            'memory_peak_gb': end_mem[1],
            'metrics': metrics,
            'under_limit': end_mem[1] < 39.0
        }
        
        print(f"DeepSpeed results: {results}")
        return results
        
    except Exception as e:
        print(f"DeepSpeed test failed: {e}")
        return {
            'variant': 'deepspeed',
            'success': False,
            'error': str(e),
            'memory_peak_gb': get_memory_usage()[1]
        }


def test_accelerate_variant() -> Dict[str, Any]:
    """Test HF Accelerate device mapping variant."""
    print("\n=== Testing HF Accelerate Device Mapping Variant ===")
    
    reset_memory_stats()
    
    config = create_test_config()
    config_loader = ConfigLoader(config)
    
    # Create trainer with Accelerate
    trainer = MemoryEfficientTrainer(
        config_loader=config_loader,
        use_bf16=True,
        use_checkpointing=True,
        use_8bit_optimizer=True,
        optimizer_type="paged_adamw8bit",
        grad_accum_steps=4,
        use_accelerate=True,
        max_gpu_memory="39GiB",
        max_cpu_memory="70GiB",
        offload_folder="./test_offload"
    )
    
    # Create simple model for testing
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(1000, 256)  # Large embedding
            self.large_linear = torch.nn.Linear(256, 1024)   # Large linear layer
            self.attention = torch.nn.MultiheadAttention(256, 8, batch_first=True)
            self.output = torch.nn.Linear(1024, 2)
            
        def forward(self, rgb, class_id, **kwargs):
            B = rgb.shape[0]
            # Use class_id for embedding
            emb = self.embeddings(class_id)  # (B, 256)
            emb = emb.unsqueeze(1).expand(-1, 10, -1)  # (B, 10, 256)
            
            # Self-attention
            attn_out, _ = self.attention(emb, emb, emb)
            
            # Process through large layers
            x = self.large_linear(attn_out.mean(1))  # (B, 1024)
            out = self.output(x)
            
            return {'p_hit': torch.sigmoid(out[:, 0:1]), 'p_kill': torch.sigmoid(out[:, 1:2])}
    
    model = SimpleTestModel()
    
    try:
        # Prepare model with Accelerate
        model, optimizer = trainer.prepare_model_for_training(model, "test_model")
        
        # Create synthetic data
        batch = create_synthetic_data(batch_size=2)
        
        # Training step
        start_mem = get_memory_usage()
        metrics = trainer.training_step(model, batch, optimizer, step=0)
        end_mem = get_memory_usage()
        
        # Check device placement
        device_info = {}
        if hasattr(model, 'hf_device_map'):
            device_info = dict(model.hf_device_map)
        
        results = {
            'variant': 'accelerate',
            'success': True,
            'memory_start_gb': start_mem[0],
            'memory_peak_gb': end_mem[1],
            'metrics': metrics,
            'device_map': device_info,
            'under_limit': end_mem[1] < 39.0
        }
        
        print(f"Accelerate results: {results}")
        return results
        
    except Exception as e:
        print(f"Accelerate test failed: {e}")
        return {
            'variant': 'accelerate',
            'success': False,
            'error': str(e),
            'memory_peak_gb': get_memory_usage()[1]
        }


def test_baseline_variant() -> Dict[str, Any]:
    """Test baseline training without optimizations."""
    print("\n=== Testing Baseline (No Optimizations) ===")
    
    reset_memory_stats()
    
    config = create_test_config()
    config_loader = ConfigLoader(config)
    
    # Create trainer without optimizations
    trainer = MemoryEfficientTrainer(
        config_loader=config_loader,
        use_bf16=False,  # FP32
        use_checkpointing=False,
        use_8bit_optimizer=False,
        grad_accum_steps=1,
        use_deepspeed=False,
        use_accelerate=False
    )
    
    # Simple model
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(256, 2)
            
        def forward(self, rgb, **kwargs):
            B = rgb.shape[0]
            x = rgb.mean([1, 2, 3, 4])  # Global average pool
            out = self.fc(x)
            return {'p_hit': torch.sigmoid(out[:, 0:1]), 'p_kill': torch.sigmoid(out[:, 1:2])}
    
    model = SimpleTestModel()
    
    try:
        # Prepare model
        model, optimizer = trainer.prepare_model_for_training(model, "test_model")
        
        # Create synthetic data
        batch = create_synthetic_data(batch_size=2)
        
        # Training step
        start_mem = get_memory_usage()
        metrics = trainer.training_step(model, batch, optimizer, step=0)
        end_mem = get_memory_usage()
        
        results = {
            'variant': 'baseline',
            'success': True,
            'memory_start_gb': start_mem[0],
            'memory_peak_gb': end_mem[1],
            'metrics': metrics,
            'under_limit': end_mem[1] < 39.0
        }
        
        print(f"Baseline results: {results}")
        return results
        
    except Exception as e:
        print(f"Baseline test failed: {e}")
        return {
            'variant': 'baseline',
            'success': False,
            'error': str(e),
            'memory_peak_gb': get_memory_usage()[1]
        }


def run_smoke_test() -> Dict[str, Any]:
    """Run complete smoke test with all variants."""
    print("🔥 TRIDENT-Net Memory Optimization Smoke Test 🔥")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test all variants
    results = {}
    
    # Test baseline
    results['baseline'] = test_baseline_variant()
    
    # Test optimized variants only if libraries are available
    try:
        import bitsandbytes
        results['deepspeed'] = test_deepspeed_variant()
    except ImportError:
        print("Skipping DeepSpeed test - bitsandbytes not available")
        results['deepspeed'] = {'variant': 'deepspeed', 'success': False, 'error': 'bitsandbytes not available'}
    
    try:
        import accelerate
        results['accelerate'] = test_accelerate_variant()
    except ImportError:
        print("Skipping Accelerate test - accelerate not available")
        results['accelerate'] = {'variant': 'accelerate', 'success': False, 'error': 'accelerate not available'}
    
    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    
    for variant, result in results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        memory = f"{result.get('memory_peak_gb', 0):.2f} GB"
        under_limit = "✅" if result.get('under_limit', False) else "❌"
        
        print(f"{variant:>12}: {status} | Peak Memory: {memory} | Under 39GB: {under_limit}")
        
        if not result.get('success', False) and 'error' in result:
            print(f"{'':>15} Error: {result['error']}")
    
    # Overall assessment
    successful_variants = [v for v in results.values() if v.get('success', False)]
    memory_efficient_variants = [v for v in successful_variants if v.get('under_limit', False)]
    
    print(f"\nSuccessful variants: {len(successful_variants)}/{len(results)}")
    print(f"Memory-efficient variants: {len(memory_efficient_variants)}/{len(results)}")
    
    if memory_efficient_variants:
        best_variant = min(memory_efficient_variants, key=lambda x: x.get('memory_peak_gb', float('inf')))
        print(f"Best variant: {best_variant['variant']} ({best_variant['memory_peak_gb']:.2f} GB)")
    
    return results


if __name__ == "__main__":
    # Clean up any previous test artifacts
    import shutil
    test_dirs = ['./test_offload', './offload']
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
    
    # Run smoke test
    results = run_smoke_test()
    
    # Save results
    results_path = Path("smoke_test_results.json")
    with open(results_path, 'w') as f:
        # Convert any tensor values to basic types for JSON serialization
        clean_results = {}
        for variant, result in results.items():
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    clean_result[k] = v.item() if v.numel() == 1 else v.tolist()
                elif isinstance(v, dict):
                    clean_result[k] = {kk: vv.item() if isinstance(vv, torch.Tensor) and vv.numel() == 1 else vv 
                                     for kk, vv in v.items()}
                else:
                    clean_result[k] = v
            clean_results[variant] = clean_result
        
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Exit with appropriate code
    all_success = all(r.get('success', False) for r in results.values())
    memory_success = any(r.get('under_limit', False) for r in results.values() if r.get('success', False))
    
    if memory_success:
        print("🎉 SUCCESS: At least one memory-efficient variant works!")
        sys.exit(0)
    elif all_success:
        print("⚠️  WARNING: All variants work but none are memory-efficient")
        sys.exit(1)
    else:
        print("💥 FAILURE: No variants work properly")
        sys.exit(2)