"""
Memory-efficient training CLI for TRIDENT-Net.

Extends the existing CLI with memory optimization options.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

from .cli import load_config, setup_logging, _device, _build_models, _create_synthetic_loaders
from .memory_efficient_trainer import MemoryEfficientTrainer
from .config import ConfigLoader
from ..data.dataset import create_data_loaders


def command_train_memory_efficient(args) -> None:
    """Execute memory-efficient training with optimization options."""
    print(f"Starting memory-efficient TRIDENT-Net training config={args.config}")
    
    # Load config
    cfg = load_config(args.config)
    config_loader = ConfigLoader(cfg)
    
    # Override data sources if provided
    if args.jsonl or args.video_root:
        data = cfg.setdefault('data', {})
        sources = data.setdefault('sources', {})
        if args.jsonl:
            sources['jsonl_path'] = args.jsonl
        if args.video_root:
            sources['video_root'] = args.video_root
    
    # Override DataLoader settings
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        cfg.setdefault('data', {}).setdefault('loader', {})['batch_size'] = args.batch_size
    if hasattr(args, 'num_workers') and args.num_workers is not None:
        cfg.setdefault('data', {}).setdefault('loader', {})['num_workers'] = args.num_workers
    if hasattr(args, 'pin_memory') and args.pin_memory is not None:
        cfg.setdefault('data', {}).setdefault('loader', {})['pin_memory'] = args.pin_memory
    
    # Data loaders
    use_synth = bool(args.synthetic)
    if (args.jsonl or args.video_root):
        use_synth = False
    if use_synth:
        print("using_synthetic=true")
        train_loader, val_loader = _create_synthetic_loaders(cfg)
    else:
        train_loader, val_loader = create_data_loaders(cfg)
    
    device = _device()
    print(f"device={device} cuda_available={torch.cuda.is_available()} "
          f"cuda_devices={torch.cuda.device_count()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Create memory-efficient trainer
    trainer = MemoryEfficientTrainer(
        config_loader=config_loader,
        device=device,
        use_bf16=args.use_bf16,
        use_checkpointing=args.checkpoint_every_layer,
        use_8bit_optimizer=args.optimizer in ["adamw8bit", "paged_adamw8bit"],
        optimizer_type=args.optimizer,
        grad_accum_steps=args.grad_accum_steps,
        use_deepspeed=args.zero_stage > 0,
        use_accelerate=args.device_map == "auto",
        max_gpu_memory=args.max_gpu_mem,
        max_cpu_memory=args.cpu_mem,
        offload_folder="./offload" if args.device_map == "auto" else None,
        use_qlora=args.qlora
    )
    
    # Build models
    models = _build_models()
    
    # Prepare fusion model for training (F2)
    f2 = models['f2']
    f2, optimizer = trainer.prepare_model_for_training(f2, "fusion_f2")
    
    print(f"Memory optimization settings:")
    print(f"  BF16: {args.use_bf16}")
    print(f"  Checkpointing: {args.checkpoint_every_layer}")
    print(f"  8-bit optimizer: {args.optimizer}")
    print(f"  Gradient accumulation: {args.grad_accum_steps}")
    print(f"  ZeRO stage: {args.zero_stage}")
    print(f"  Device map: {args.device_map}")
    print(f"  Max GPU memory: {args.max_gpu_mem}")
    print(f"  QLoRA: {args.qlora}")
    
    # Training parameters
    epochs = cfg.get('training', {}).get('epochs', {}).get('train_fusion', 3)
    
    # Training loop
    print(f"Training for {epochs} epochs with memory optimizations...")
    
    best_val_metric = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training epoch
        f2.train()
        train_metrics = {'loss': 0.0, 'count': 0}
        
        for step, batch in enumerate(train_loader):
            # Move batch to device (if not using accelerate)
            if not trainer.use_accelerate:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Training step
            step_metrics = trainer.training_step(f2, batch, optimizer, step)
            
            # Accumulate metrics
            train_metrics['loss'] += step_metrics['loss']
            train_metrics['count'] += 1
            
            # Log memory usage periodically
            if step % 10 == 0:
                trainer.log_memory_usage(step, f"Epoch {epoch+1} Step {step}: ")
                print(f"Step {step}: loss={step_metrics['loss']:.4f}")
        
        # Average training metrics
        avg_train_loss = train_metrics['loss'] / max(train_metrics['count'], 1)
        print(f"Train loss: {avg_train_loss:.4f}")
        
        # Validation
        f2.eval()
        val_metrics = {'loss': 0.0, 'count': 0}
        preds_h, preds_k, tgts_h, tgts_k = [], [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                if not trainer.use_accelerate:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                with torch.autocast(device_type="cuda", dtype=trainer.mixed_precision_dtype, 
                                  enabled=trainer.use_bf16):
                    outputs = f2(**batch)
                    
                    # Collect predictions (this is simplified - would need actual model outputs)
                    if hasattr(outputs, 'p_hit'):
                        preds_h.append(outputs.p_hit.cpu())
                    if hasattr(outputs, 'p_kill'):
                        preds_k.append(outputs.p_kill.cpu())
                
                # Collect targets
                if 'labels' in batch:
                    if 'hit' in batch['labels']:
                        tgts_h.append(batch['labels']['hit'].cpu())
                    if 'kill' in batch['labels']:
                        tgts_k.append(batch['labels']['kill'].cpu())
        
        # Compute validation metrics
        if preds_h and tgts_h:
            from ..common.metrics import auroc, f1
            ph = torch.cat(preds_h)
            th = torch.cat(tgts_h)
            try:
                val_au = auroc(th, ph)
            except:
                val_au = float('nan')
            val_f1 = f1(th, ph)
            print(f"Validation AUROC: {val_au:.3f} F1: {val_f1:.3f}")
            
            # Update best metric
            metric = float(val_au) if not (val_au != val_au) else float(val_f1)
            if metric > best_val_metric:
                best_val_metric = metric
                print(f"New best validation metric: {best_val_metric:.4f}")
        
        # Log final memory usage for epoch
        trainer.log_memory_usage(epoch, f"End of epoch {epoch+1}: ")
    
    print(f"\nTraining completed. Best validation metric: {best_val_metric:.4f}")
    
    # Final memory report
    memory_stats = trainer.get_memory_stats()
    print(f"\nFinal memory usage:")
    for key, value in memory_stats.items():
        print(f"  {key}: {value:.2f} GB")
    
    # Check if we stayed under memory limit
    if 'gpu_max_memory_allocated' in memory_stats:
        max_gpu_gb = memory_stats['gpu_max_memory_allocated']
        target_gb = 39.0  # A100-40GB target
        if max_gpu_gb < target_gb:
            print(f"✅ SUCCESS: Peak GPU memory ({max_gpu_gb:.1f} GB) < target ({target_gb} GB)")
        else:
            print(f"❌ WARNING: Peak GPU memory ({max_gpu_gb:.1f} GB) > target ({target_gb} GB)")


def add_memory_efficient_args(parser: argparse.ArgumentParser) -> None:
    """Add memory optimization arguments to parser."""
    
    # Memory optimization flags
    mem_group = parser.add_argument_group('Memory Optimization')
    mem_group.add_argument('--use-bf16', action='store_true', default=True,
                          help='Enable BF16 mixed precision (default: True)')
    mem_group.add_argument('--no-bf16', dest='use_bf16', action='store_false',
                          help='Disable BF16 mixed precision')
    
    mem_group.add_argument('--checkpoint-every-layer', action='store_true', default=True,
                          help='Enable activation checkpointing for heavy layers (default: True)')
    mem_group.add_argument('--no-checkpointing', dest='checkpoint_every_layer', action='store_false',
                          help='Disable activation checkpointing')
    
    # Gradient accumulation
    mem_group.add_argument('--grad-accum-steps', type=int, default=8,
                          help='Gradient accumulation steps for micro-batching (default: 8)')
    
    # Optimizer options
    mem_group.add_argument('--optimizer', choices=['adamw', 'adamw8bit', 'paged_adamw8bit'], 
                          default='adamw8bit',
                          help='Optimizer type (default: adamw8bit)')
    
    # DeepSpeed options
    ds_group = parser.add_argument_group('DeepSpeed ZeRO')
    ds_group.add_argument('--zero-stage', type=int, choices=[0, 1, 2, 3], default=2,
                         help='DeepSpeed ZeRO stage (0=disabled, default: 2)')
    ds_group.add_argument('--deepspeed-config', type=str, default='ds_config.json',
                         help='DeepSpeed configuration file (default: ds_config.json)')
    
    # Accelerate options
    acc_group = parser.add_argument_group('HF Accelerate')
    acc_group.add_argument('--device-map', choices=['auto', 'balanced', 'custom'], default='auto',
                          help='Device mapping strategy (default: auto)')
    acc_group.add_argument('--max-gpu-mem', type=str, default='39GiB',
                          help='Maximum GPU memory allocation (default: 39GiB)')
    acc_group.add_argument('--cpu-mem', type=str, default='70GiB',
                          help='Maximum CPU memory for offload (default: 70GiB)')
    
    # QLoRA options
    qlora_group = parser.add_argument_group('QLoRA')
    qlora_group.add_argument('--qlora', action='store_true', default=False,
                            help='Enable QLoRA for Transformer blocks (default: False)')


def create_memory_efficient_parser() -> argparse.ArgumentParser:
    """Create argument parser with memory-efficient training options."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient training for TRIDENT-Net",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument('--config', default='tasks.yml', help='Configuration file')
    parser.add_argument('--synthetic', action='store_true', default=False,
                       help='Use synthetic data instead of real dataset')
    parser.add_argument('--jsonl', default=None, help='Override data.sources.jsonl_path')
    parser.add_argument('--video-root', dest='video_root', default=None, 
                       help='Override data.sources.video_root')
    
    # DataLoader options
    loader_group = parser.add_argument_group('DataLoader')
    loader_group.add_argument('--batch-size', type=int, default=None,
                            help='Override data.loader.batch_size')
    loader_group.add_argument('--num-workers', type=int, default=None,
                            help='Override data.loader.num_workers')
    
    pm_group = loader_group.add_mutually_exclusive_group()
    pm_group.add_argument('--pin-memory', dest='pin_memory', action='store_true',
                         help='Enable DataLoader pin_memory')
    pm_group.add_argument('--no-pin-memory', dest='pin_memory', action='store_false',
                         help='Disable DataLoader pin_memory')
    parser.set_defaults(pin_memory=None)
    
    # Add memory optimization arguments
    add_memory_efficient_args(parser)
    
    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    return parser


def main_memory_efficient():
    """Main entry point for memory-efficient training."""
    parser = create_memory_efficient_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute memory-efficient training
    command_train_memory_efficient(args)


if __name__ == "__main__":
    main_memory_efficient()