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
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .cli import load_config, setup_logging, _device, _build_models, _create_synthetic_loaders, _kin_aug
from .memory_efficient_trainer import MemoryEfficientTrainer
from .config import ConfigLoader
from ..data.dataset import create_data_loaders


def command_train_memory_efficient(args) -> None:
    """Execute memory-efficient training with optimization options."""
    print(f"Starting memory-efficient TRIDENT-Net training config={args.config}")
    
    # Load config
    cfg = load_config(args.config)
    config_loader = ConfigLoader()
    config_loader.config = cfg
    
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
        # Also update the raw_config so the trainer can access it
        config_loader.raw_config.setdefault('data', {}).setdefault('loader', {})['batch_size'] = args.batch_size
    if hasattr(args, 'num_workers') and args.num_workers is not None:
        cfg.setdefault('data', {}).setdefault('loader', {})['num_workers'] = args.num_workers
        config_loader.raw_config.setdefault('data', {}).setdefault('loader', {})['num_workers'] = args.num_workers
    if hasattr(args, 'pin_memory') and args.pin_memory is not None:
        cfg.setdefault('data', {}).setdefault('loader', {})['pin_memory'] = args.pin_memory
        config_loader.raw_config.setdefault('data', {}).setdefault('loader', {})['pin_memory'] = args.pin_memory
    
    # Debug: Log the configuration being used
    print(f"Configuration after overrides: batch_size={cfg.get('data', {}).get('loader', {}).get('batch_size', 'not set')}")
    print(f"Raw config batch_size={config_loader.raw_config.get('data', {}).get('loader', {}).get('batch_size', 'not set')}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    
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
    
    # CPU compatibility checks and auto-adjustments
    is_cpu_only = not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    if is_cpu_only:
        print(f"🖥️  CPU-only mode detected. Adjusting settings for compatibility:")
        if args.use_fp16:
            print(f"   • Disabling FP16 (not supported on CPU) → FP32")
            args.use_fp16 = False
        if args.optimizer in ["adamw8bit", "paged_adamw8bit"]:
            print(f"   • Disabling 8-bit optimizer (requires CUDA) → adamw")
            args.optimizer = "adamw"
        if args.zero_stage > 0:
            print(f"   • Disabling DeepSpeed ZeRO (requires CUDA) → stage 0")
            args.zero_stage = 0
        if args.device_map == "auto":
            print(f"   • Disabling Accelerate device mapping (not needed on CPU)")
            args.device_map = "balanced"
        print(f"   ✅ CPU-compatible configuration applied")
    
    # Resolve configuration conflicts for CUDA systems
    elif torch.cuda.is_available():
        conflicts_resolved = []
        if args.zero_stage > 0 and args.optimizer in ["adamw8bit", "paged_adamw8bit"]:
            print(f"🔧 Resolving conflict: DeepSpeed ZeRO + 8-bit optimizer")
            print(f"   • DeepSpeed will handle optimizer → disabling 8-bit optimizer")
            args.optimizer = "adamw"
            conflicts_resolved.append("8-bit optimizer → DeepSpeed optimizer")
            
        if args.zero_stage > 0 and args.device_map == "auto":
            print(f"🔧 Resolving conflict: DeepSpeed ZeRO + Accelerate device mapping")
            print(f"   • Prioritizing DeepSpeed → disabling Accelerate device mapping")
            args.device_map = "balanced"
            conflicts_resolved.append("Accelerate device map → DeepSpeed handling")
            
        if conflicts_resolved:
            print(f"   ✅ Resolved {len(conflicts_resolved)} configuration conflicts")
    
    # Validate DeepSpeed config file if specified
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config and args.deepspeed_config != 'ds_config.json':
        from pathlib import Path
        config_path = Path(args.deepspeed_config)
        if not config_path.exists():
            print(f"⚠️  WARNING: DeepSpeed config file not found: {args.deepspeed_config}")
            print(f"   Falling back to automatic configuration generation")
            args.deepspeed_config = None
        elif is_cpu_only and 'cpu_only' not in str(config_path):
            print(f"⚠️  WARNING: Using GPU-optimized config on CPU system: {args.deepspeed_config}")
            print(f"   Consider using configs/cpu_only_30gb_ram.json for CPU-only systems")
    
    # Create memory-efficient trainer AFTER CPU compatibility adjustments
    trainer = MemoryEfficientTrainer(
        config_loader=config_loader,
        device=device,
        use_fp16=args.use_fp16,
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
    
    # For memory-efficient training, we'll preprocess the data to match f2 expectations
    # and train only the fusion model (f2) with preprocessed features
    f2 = models['f2']
    
    # Move branch models to device for preprocessing (but don't train them)
    for name, model in models.items():
        if name != 'f2':  # Don't move f2 yet, it will be handled by trainer
            # Always move branch models to device, regardless of Accelerate usage
            # since they're used for preprocessing and aren't managed by Accelerate
            model = model.to(device)
            model.eval()  # Set to eval mode since we're not training these
            models[name] = model  # Update the model in the dict
    
    # Prepare fusion model for training
    f2, optimizer = trainer.prepare_model_for_training(f2, "fusion_f2")
    
    print(f"Memory optimization settings:")
    print(f"  FP16: {args.use_fp16}")
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
    
    # Preprocessing function to convert raw batch to fusion input format
    def preprocess_batch_for_fusion(batch, models, device):
        """Convert raw batch data to format expected by fusion model f2."""
        with torch.no_grad():
            # Move inputs to device 
            rgb = batch['rgb'].to(device)  # [B,3,T,H,W]
            ir = batch['ir'].to(device)    # [B,1,T,H,W] 
            kin = batch['kin'].to(device)  # [B,3,9]
            class_ids = batch.get('class_id', torch.zeros(rgb.shape[0], dtype=torch.long)).to(device)
            
            # Ensure all branch models are on the correct device
            for name, model in models.items():
                if name != 'f2':
                    model_device = next(model.parameters()).device
                    if model_device != device:
                        print(f"⚠️  Moving {name} from {model_device} to {device}")
                        models[name] = model.to(device)
            
            # Process through branch models (no gradients needed)
            try:
                i1_out = models['i1'](rgb)
                t1_out = models['t1'](ir)
                r1_feats, _ = models['r1'](kin)
                
                # Kinematics augmentation (from _kin_aug function)
                k_aug, k_tokens = _kin_aug(kin, r1_feats)
                zr2, _ = models['r2'](k_aug)
                zr3, _ = models['r3'](k_tokens)
                
            except Exception as e:
                print(f"❌ Error in branch model processing: {e}")
                print(f"   RGB shape: {rgb.shape}, device: {rgb.device}")
                print(f"   IR shape: {ir.shape}, device: {ir.device}")
                print(f"   KIN shape: {kin.shape}, device: {kin.device}")
                for name, model in models.items():
                    if name != 'f2':
                        try:
                            model_device = next(model.parameters()).device
                            print(f"   {name} device: {model_device}")
                        except:
                            print(f"   {name} device: unknown")
                raise
            
            # Concat features per architecture specification
            zi = torch.cat([i1_out['zi'], torch.zeros(rgb.shape[0], 256, device=device)], dim=1)  # 768
            zt = t1_out['zt']  # 256
            zt = torch.cat([zt, torch.zeros_like(zt)], dim=1)  # Pad to 512
            zr = torch.cat([zr2, zr3], dim=1)  # 384 (192+192)
            
            # Prepare fusion inputs
            fusion_batch = {
                'zi': zi,
                'zt': zt, 
                'zr': zr,
                'class_ids': class_ids,
                'events': None,  # Events would be collected from branch outputs
                'labels': batch.get('labels')  # Pass through labels for loss computation
            }
            
            return fusion_batch
    
    best_val_metric = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training epoch
        f2.train()
        train_metrics = {'loss': 0.0, 'count': 0}
        
        for step, batch in enumerate(train_loader):
            # Preprocess batch to fusion format
            fusion_batch = preprocess_batch_for_fusion(batch, models, device)
            
            # Training step on preprocessed batch
            step_metrics = trainer.training_step(f2, fusion_batch, optimizer, step)
            
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
                # Preprocess batch to fusion format
                fusion_batch = preprocess_batch_for_fusion(batch, models, device)
                
                # Forward pass
                with torch.autocast(device_type="cuda", dtype=trainer.mixed_precision_dtype, 
                                  enabled=trainer.use_fp16):
                    outputs = f2(**{k: v for k, v in fusion_batch.items() if k != 'labels'})
                    
                    # Collect predictions
                    if isinstance(outputs, tuple):
                        z_fused, p_hit, p_kill = outputs[:3]
                        preds_h.append(p_hit.cpu())
                        preds_k.append(p_kill.cpu())
                    elif isinstance(outputs, dict):
                        if 'p_hit' in outputs:
                            preds_h.append(outputs['p_hit'].cpu())
                        if 'p_kill' in outputs:
                            preds_k.append(outputs['p_kill'].cpu())
                    elif hasattr(outputs, 'p_hit'):
                        preds_h.append(outputs.p_hit.cpu())
                        if hasattr(outputs, 'p_kill'):
                            preds_k.append(outputs.p_kill.cpu())
                
                # Collect targets
                if fusion_batch.get('labels') is not None:
                    labels = fusion_batch['labels']
                    if isinstance(labels, dict):
                        if 'hit' in labels:
                            tgts_h.append(labels['hit'].cpu())
                        if 'kill' in labels:
                            tgts_k.append(labels['kill'].cpu())
                    else:
                        # Handle tensor labels 
                        if labels.dim() == 2 and labels.shape[1] >= 2:
                            tgts_h.append(labels[:, 0].cpu())
                            tgts_k.append(labels[:, 1].cpu())
                        else:
                            tgts_h.append(labels.cpu())
        
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
    mem_group.add_argument('--use-fp16', action='store_true', default=True,
                          help='Enable FP16 mixed precision (default: True)')
    mem_group.add_argument('--no-fp16', dest='use_fp16', action='store_false',
                          help='Disable FP16 mixed precision')
    
    mem_group.add_argument('--checkpoint-every-layer', action='store_true', default=True,
                          help='Enable activation checkpointing for heavy layers (default: True)')
    mem_group.add_argument('--no-checkpointing', dest='checkpoint_every_layer', action='store_false',
                          help='Disable activation checkpointing')
    
    # Gradient accumulation
    mem_group.add_argument('--grad-accum-steps', type=int, default=4,
                          help='Gradient accumulation steps for micro-batching (default: 4)')
    
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
                          help='Maximum CPU memory for offload (default: 70GiB for training, 30GiB for evaluation)')
    
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