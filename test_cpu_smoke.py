"""
Simplified smoke test for memory-efficient TRIDENT-Net training.

Tests the basic functionality without requiring GPU.
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trident.runtime.memory_efficient_trainer import MemoryEfficientTrainer
from trident.runtime.config import ConfigLoader
from trident.common.sdpa_attention import SDPAAttention, convert_attention_to_sdpa


def create_test_config() -> Dict[str, Any]:
    """Create minimal test configuration."""
    return {
        'environment': {
            'seed': 12345,
            'device': 'auto'
        },
        'data': {
            'loader': {
                'batch_size': 1,
                'num_workers': 0,
                'pin_memory': False
            }
        },
        'training': {
            'optimizer': {
                'type': 'adamw',
                'lr': 2e-4,
                'weight_decay': 0.01
            },
            'amp': False,  # Disable AMP for CPU
            'grad_clip_norm': 1.0
        }
    }


def test_memory_efficient_trainer():
    """Test basic functionality of MemoryEfficientTrainer."""
    print("Testing MemoryEfficientTrainer...")
    
    config = create_test_config()
    config_loader = ConfigLoader()
    config_loader.config = config  # Set config directly for testing
    config_loader.raw_config = config
    
    # Test with CPU-friendly settings
    trainer = MemoryEfficientTrainer(
        config_loader=config_loader,
        use_bf16=False,  # CPU doesn't support bf16
        use_checkpointing=True,
        use_8bit_optimizer=False,  # Not available on CPU
        optimizer_type="adamw8bit",
        grad_accum_steps=2,
        use_deepspeed=False,  # Disable for CPU test
        use_accelerate=False   # Disable for CPU test
    )
    
    # Simple test model
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
            
        def forward(self, x, **kwargs):
            out = self.linear(x)
            return {'p_hit': torch.sigmoid(out[:, 0:1]), 'p_kill': torch.sigmoid(out[:, 1:2])}
    
    model = TestModel()
    
    # Test model preparation
    prepared_model, optimizer = trainer.prepare_model_for_training(model, "test")
    
    # Test training step
    batch = {
        'x': torch.randn(2, 10),
        'labels': {'hit': torch.rand(2, 1), 'kill': torch.rand(2, 1)}
    }
    
    metrics = trainer.training_step(prepared_model, batch, optimizer, 0)
    
    print(f"✅ MemoryEfficientTrainer test passed. Metrics: {metrics}")
    return True


def test_sdpa_attention():
    """Test SDPA attention module."""
    print("Testing SDPA attention...")
    
    # Test basic SDPA attention
    attn = SDPAAttention(embed_dim=64, num_heads=4, dropout=0.1)
    
    # Test forward pass
    B, L, E = 2, 10, 64
    query = torch.randn(B, L, E)
    key = torch.randn(B, L, E)
    value = torch.randn(B, L, E)
    
    output, weights = attn(query, key, value, need_weights=True)
    
    assert output.shape == (B, L, E), f"Expected {(B, L, E)}, got {output.shape}"
    print(f"✅ SDPA attention test passed. Output shape: {output.shape}")
    
    # Test conversion function
    class SimpleTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(64, 4, batch_first=True)
            self.linear = torch.nn.Linear(64, 64)
    
    model = SimpleTransformer()
    print(f"Before conversion: {type(model.attention)}")
    
    convert_attention_to_sdpa(model)
    print(f"After conversion: {type(model.attention)}")
    
    print("✅ SDPA conversion test passed")
    return True


def test_memory_optimization_flags():
    """Test different memory optimization combinations."""
    print("Testing memory optimization flag combinations...")
    
    config = create_test_config()
    config_loader = ConfigLoader()
    config_loader.config = config  # Set config directly for testing
    config_loader.raw_config = config
    
    test_configs = [
        {"use_bf16": False, "use_checkpointing": True, "use_8bit_optimizer": False},
        {"use_bf16": False, "use_checkpointing": False, "use_8bit_optimizer": False},
        {"use_bf16": False, "use_checkpointing": True, "grad_accum_steps": 4},
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"  Testing config {i+1}: {test_config}")
        
        trainer = MemoryEfficientTrainer(
            config_loader=config_loader,
            **test_config
        )
        
        # Simple test
        model = torch.nn.Linear(10, 2)
        prepared_model, optimizer = trainer.prepare_model_for_training(model, f"test_{i}")
        
        print(f"    ✅ Config {i+1} passed")
    
    print("✅ All memory optimization flag tests passed")
    return True


def test_cli_args_parsing():
    """Test CLI argument integration."""
    print("Testing CLI argument parsing...")
    
    from trident.runtime.memory_efficient_cli import create_memory_efficient_parser
    
    parser = create_memory_efficient_parser()
    
    # Test various argument combinations
    test_args = [
        ["--use-bf16", "--grad-accum-steps", "8"],
        ["--no-bf16", "--optimizer", "adamw", "--zero-stage", "0"],
        ["--checkpoint-every-layer", "--device-map", "auto", "--max-gpu-mem", "39GiB"],
        ["--qlora", "--optimizer", "paged_adamw8bit"]
    ]
    
    for i, args in enumerate(test_args):
        print(f"  Testing args {i+1}: {args}")
        parsed = parser.parse_args(args)
        print(f"    ✅ Args {i+1} parsed successfully")
    
    print("✅ CLI argument parsing tests passed")
    return True


def run_cpu_smoke_test():
    """Run smoke test suitable for CPU environment."""
    print("🔥 TRIDENT-Net Memory Optimization CPU Smoke Test 🔥")
    print(f"Device: CPU (CUDA available: {torch.cuda.is_available()})")
    
    tests = [
        ("MemoryEfficientTrainer", test_memory_efficient_trainer),
        ("SDPA Attention", test_sdpa_attention),
        ("Memory Optimization Flags", test_memory_optimization_flags),
        ("CLI Arguments", test_cli_args_parsing)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = {"success": True, "error": None}
            if success:
                passed += 1
        except Exception as e:
            results[test_name] = {"success": False, "error": str(e)}
            print(f"❌ {test_name} failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("CPU SMOKE TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{test_name:>25}: {status}")
        if not result["success"]:
            print(f"{'':>27} Error: {result['error']}")
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All CPU tests passed! Memory optimization framework is working.")
    else:
        print("⚠️ Some tests failed. Check implementations.")
    
    return results


if __name__ == "__main__":
    results = run_cpu_smoke_test()
    
    # Save results
    with open("cpu_smoke_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    all_passed = all(r["success"] for r in results.values())
    sys.exit(0 if all_passed else 1)