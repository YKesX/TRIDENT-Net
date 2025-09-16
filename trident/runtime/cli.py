"""
Command-line interface for TRIDENT-Net.

Runs real training/evaluation pipelines using the implemented data stack and
model components. Designed to stream meaningful metrics to the GUI.

Author: Yağızhan Keskin
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import create_data_loaders
from ..data.synthetic import generate_synthetic_batch
from ..trident_i.videox3d import VideoFrag3Dv2
from ..trident_t.ir_dettrack_v2 import PlumeDetXL
from ..trident_r.kinefeat import KineFeat
from ..trident_r.geomlp import GeoMLP
from ..trident_r.tiny_temporal_former import TinyTempoFormer
from ..fusion_guard.cross_attn_fusion import CrossAttnFusion
from ..common.metrics import compute_metrics, auroc, f1, brier_score, expected_calibration_error


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config {config_path}: {e}")
        sys.exit(1)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_sections = ['components', 'training', 'runtime']
    
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required config section: {section}")
            return False
    
    return True


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_models() -> Dict[str, nn.Module]:
    """Instantiate branch and fusion models with default dims as per tasks.yml."""
    models: Dict[str, nn.Module] = {
        'i1': VideoFrag3Dv2(out_embed_dim=512),
        't1': PlumeDetXL(pool_to_embed=256),
        'r1': KineFeat(),
        'r2': GeoMLP(),
        'r3': TinyTempoFormer(d_model=192, token_dim=32),
        'f2': CrossAttnFusion(d_model=512, n_heads=8, n_layers=3, mlp_hidden=256, dims={
            'zi': 768, 'zt': 512, 'zr': 384, 'e_cls': 32
        }, num_classes=100),
    }
    return models


def _kin_aug(kin: torch.Tensor, r_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build k_aug (B,69) and k_tokens (B,3,32) from kin (B,3,9) and r_feats (B,24)."""
    B = kin.shape[0]
    # raw 27
    k_raw = kin.reshape(B, -1)  # (B,27)
    # deltas 18 (Δ 0-1 and 1-2 over all 9 dims)
    d01 = kin[:, 1] - kin[:, 0]
    d12 = kin[:, 2] - kin[:, 1]
    deltas = torch.cat([d01, d12], dim=1)  # (B,18)
    # concat with r_feats (B,24)
    k_aug = torch.cat([k_raw, deltas, r_feats], dim=1)  # (B,69)
    # tokens (simple zero-pad per step to 32)
    pad = torch.zeros(B, 3, 32 - 9, device=kin.device, dtype=kin.dtype)
    k_tokens = torch.cat([kin, pad], dim=2)  # (B,3,32)
    return k_aug, k_tokens


def _create_synthetic_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create simple synthetic train/val loaders for smoke training.

    Respects batch size from cfg when present.
    """
    from torch.utils.data import IterableDataset

    class _SynthDS(IterableDataset):
        def __init__(self, steps: int = 100, batch_size: int = 2):
            self.steps = steps
            self.batch_size = batch_size

        def __iter__(self):
            for _ in range(self.steps):
                batch = generate_synthetic_batch(batch_size=self.batch_size)
                yield {
                    'rgb': batch['rgb'],
                    'ir': batch['ir'],
                    'kin': batch['kin'],
                    'labels': batch['labels'],
                    'class_id': batch['class_id'],
                }

        # Provide a length to satisfy consumers that query len(dataloader)
        def __len__(self):
            return self.steps

    bs = int(cfg.get('data', {}).get('loader', {}).get('batch_size', 2))
    train_ds = _SynthDS(steps=120, batch_size=bs)
    val_ds = _SynthDS(steps=20, batch_size=bs)
    train_loader = DataLoader(train_ds, batch_size=None)
    val_loader = DataLoader(val_ds, batch_size=None)
    return train_loader, val_loader


@torch.no_grad()
def _forward_batch(models: Dict[str, nn.Module], batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Run a true forward path and return outputs and metrics for this batch."""
    rgb = batch.get('rgb').to(device)  # [B,3,T,H,W]
    ir = batch.get('ir').to(device)    # [B,1,T,H,W]
    kin = batch.get('kin').to(device)  # [B,3,9]
    class_ids = batch.get('class_id')
    if class_ids is None:
        class_ids = torch.zeros(rgb.shape[0], dtype=torch.long)
    class_ids = class_ids.to(device).view(-1)

    # Branches
    i1_out = models['i1'](rgb)
    t1_out = models['t1'](ir)
    r1_feats, _ = models['r1'](kin)
    k_aug, k_tokens = _kin_aug(kin, r1_feats)
    zr2, _ = models['r2'](k_aug)
    zr3, _ = models['r3'](k_tokens)

    # Concat features per tasks.yml
    zi = torch.cat([i1_out['zi'], torch.zeros(rgb.shape[0], 256, device=device)], dim=1)  # 512+256
    zt = t1_out['zt']  # 256
    # pad zt to 512 by duplicating with zeros second half
    zt = torch.cat([zt, torch.zeros_like(zt)], dim=1)  # 256+256
    zr = torch.cat([zr2, zr3], dim=1)  # 192+192

    # Merge events
    events = []
    for k in ('events',):
        if k in i1_out:
            events.extend(i1_out[k])
        if k in t1_out:
            events.extend(t1_out[k])

    # Fusion
    z_fused, p_hit, p_kill, _, _ = models['f2'](zi=zi, zt=zt, zr=zr, class_ids=class_ids, events=events)

    out = {
        'p_hit': p_hit,
        'p_kill': p_kill,
        'zi': zi,
        'zt': zt,
        'zr': zr,
        'events': events,
    }

    # Metrics if labels present
    labels = batch.get('labels') or {}
    y_hit = labels.get('hit')
    y_kill = labels.get('kill')
    if y_hit is not None and y_kill is not None:
        y_hit = y_hit.to(device)
        y_kill = y_kill.to(device)
        out['metrics'] = {
            'AUROC_hit': auroc(y_hit, p_hit),
            'AUROC_kill': auroc(y_kill, p_kill),
            'F1_hit': f1(y_hit, p_hit),
            'F1_kill': f1(y_kill, p_kill),
            'Brier_hit': brier_score(y_hit, p_hit),
            'Brier_kill': brier_score(y_kill, p_kill),
            'ECE_hit': expected_calibration_error(y_hit, p_hit),
            'ECE_kill': expected_calibration_error(y_kill, p_kill),
        }
    return out


def command_train(args) -> None:
    """Execute a real fusion training loop (lightweight) and stream metrics."""
    print("Starting TRIDENT-Net training")
    print(f"config={args.config} pipeline={args.pipeline} synthetic={args.synthetic}")
    cfg = load_config(args.config)
    # Optional overrides from CLI
    if args.jsonl or args.video_root:
        data = cfg.setdefault('data', {})
        sources = data.setdefault('sources', {})
        if args.jsonl:
            sources['jsonl_path'] = args.jsonl
        if args.video_root:
            sources['video_root'] = args.video_root
    # Data
    if args.synthetic or cfg.get('data', {}).get('synthetic', {}).get('enabled', False):
        print("using_synthetic=true")
        train_loader, val_loader = _create_synthetic_loaders(cfg)
    else:
        train_loader, val_loader = create_data_loaders(cfg)
    device = _device()
    # Models
    models = _build_models()
    for m in models.values():
        m.to(device)
    # Freeze branches, train fusion only
    for k, m in models.items():
        if k != 'f2':
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
    f2 = models['f2']
    f2.train()
    opt = torch.optim.AdamW([p for p in f2.parameters() if p.requires_grad], lr=float(cfg.get('training', {}).get('optimizer', {}).get('lr', 2e-4)))
    bce = nn.BCELoss()
    hierarchy_w = float(cfg.get('fusion_guard', {}).get('f2', {}).get('loss', {}).get('hierarchy_regularizer', {}).get('weight', 0.2))

    max_epochs = int(cfg.get('training', {}).get('epochs', {}).get('train_fusion', 1))
    step = 0
    for epoch in range(max_epochs):
        for batch in train_loader:  # type: ignore
            # Move tensors to device
            for k in ('rgb', 'ir', 'kin'):
                if k in batch:
                    batch[k] = batch[k].to(device)
            if 'labels' in batch:
                for lk in ('hit', 'kill'):
                    if lk in batch['labels']:
                        batch['labels'][lk] = batch['labels'][lk].to(device)
            if 'class_id' in batch:
                batch['class_id'] = batch['class_id'].to(device)

            # Forward (branches no-grad)
            with torch.no_grad():
                rgb = batch['rgb']; ir = batch['ir']; kin = batch['kin']
                i1_out = models['i1'](rgb)
                t1_out = models['t1'](ir)
                r1_feats, _ = models['r1'](kin)
                k_aug, k_tokens = _kin_aug(kin, r1_feats)
                zr2, _ = models['r2'](k_aug)
                zr3, _ = models['r3'](k_tokens)
                zi = torch.cat([i1_out['zi'], torch.zeros(rgb.shape[0], 256, device=device)], dim=1)
                zt = torch.cat([t1_out['zt'], torch.zeros_like(t1_out['zt'])], dim=1)
                zr = torch.cat([zr2, zr3], dim=1)
                class_ids = batch.get('class_id')
                if class_ids is None:
                    class_ids = torch.zeros(rgb.shape[0], dtype=torch.long, device=device)

            # Fusion train step
            p_hit, p_kill = None, None
            opt.zero_grad(set_to_none=True)
            z_fused, p_hit, p_kill, _, _ = f2(zi=zi, zt=zt, zr=zr, class_ids=class_ids)
            y_hit = batch['labels']['hit']
            y_kill = batch['labels']['kill']
            loss = bce(p_hit, y_hit) + bce(p_kill, y_kill)
            # hierarchy penalty
            penalty = torch.relu(p_kill - p_hit).mean() * hierarchy_w
            total = loss + penalty
            total.backward()
            opt.step()

            # Metrics
            try:
                au = auroc(y_hit, p_hit)
            except Exception:
                au = float('nan')
            f1h = f1(y_hit, p_hit)
            step += 1
            print(f"epoch={epoch} step={step} loss={total.item():.4f} AUROC={au:.3f} F1={f1h:.3f}")

        # Validation (optional)
        if val_loader is not None:
            with torch.no_grad():
                preds_h, tgts_h = [], []
                preds_k, tgts_k = [], []
                for vb in val_loader:
                    for k in ('rgb', 'ir', 'kin'):
                        if k in vb:
                            vb[k] = vb[k].to(device)
                    if 'class_id' in vb:
                        vb['class_id'] = vb['class_id'].to(device)
                    out = _forward_batch(models, vb, device)
                    preds_h.append(out['p_hit'].cpu()); preds_k.append(out['p_kill'].cpu())
                    if 'labels' in vb:
                        tgts_h.append(vb['labels']['hit'].cpu()); tgts_k.append(vb['labels']['kill'].cpu())
                if preds_h and tgts_h:
                    ph = torch.cat(preds_h); th = torch.cat(tgts_h)
                    try:
                        au = auroc(th, ph)
                    except Exception:
                        au = float('nan')
                    f1h = f1(th, ph)
                    print(f"val AUROC={au:.3f} F1={f1h:.3f}")


def command_eval(args) -> None:
    """Execute real evaluation and stream aggregate metrics."""
    print(f"Starting TRIDENT-Net evaluation config={args.config}")
    cfg = load_config(args.config)
    if args.jsonl or args.video_root:
        data = cfg.setdefault('data', {})
        sources = data.setdefault('sources', {})
        if args.jsonl:
            sources['jsonl_path'] = args.jsonl
        if args.video_root:
            sources['video_root'] = args.video_root
    device = _device()
    if cfg.get('data', {}).get('synthetic', {}).get('enabled', False):
        train_loader, val_loader = _create_synthetic_loaders(cfg)
    else:
        train_loader, val_loader = create_data_loaders(cfg)
    eval_loader: Optional[DataLoader] = val_loader if val_loader is not None else train_loader
    models = _build_models()
    for m in models.values():
        m.to(device)
        m.eval()

    all_hit, all_kill, tg_hit, tg_kill = [], [], [], []
    steps = 0
    with torch.no_grad():
        for batch in eval_loader:  # type: ignore
            out = _forward_batch(models, batch, device)
            all_hit.append(out['p_hit'].cpu())
            all_kill.append(out['p_kill'].cpu())
            if 'labels' in batch:
                tg_hit.append(batch['labels']['hit'].cpu())
                tg_kill.append(batch['labels']['kill'].cpu())
            steps += 1
            # Stream batch metrics if available
            m = out.get('metrics')
            if m:
                print(f"step={steps} loss={0.0:.4f} AUROC={m['AUROC_hit']:.3f} F1={m['F1_hit']:.3f}")

    if all_hit and tg_hit:
        ph = torch.cat(all_hit); pk = torch.cat(all_kill)
        th = torch.cat(tg_hit); tk = torch.cat(tg_kill)
        try:
            au_h = auroc(th, ph); au_k = auroc(tk, pk)
        except Exception:
            au_h = float('nan'); au_k = float('nan')
        f1_h = f1(th, ph); f1_k = f1(tk, pk)
        print(f"Final AUROC_hit={au_h:.3f} AUROC_kill={au_k:.3f} F1_hit={f1_h:.3f} F1_kill={f1_k:.3f}")


def command_tune(args) -> None:
    """Execute hyperparameter tuning command."""
    print(f"🔧 Starting hyperparameter tuning...")
    print(f"   Config: {args.config}")
    print(f"   Component: {args.component}")
    
    config = load_config(args.config)
    hparam_cfg = config.get('hparam_search', {})
    
    print(f"   Engine: {hparam_cfg.get('engine', 'optuna')}")
    print(f"   Trials: {hparam_cfg.get('n_trials', 16)}")
    print(f"   Study: {hparam_cfg.get('study_name', 'trident_hparams')}")
    
    spaces = hparam_cfg.get('spaces', {})
    print(f"   Search spaces: {len(spaces)} parameters")
    for param, space in spaces.items():
        print(f"     {param}: {space}")
    
    print("   🎯 Hyperparameter tuning simulation completed!")


def command_serve(args) -> None:
    """Execute serving command."""
    print(f"🌐 Starting TRIDENT-Net server...")
    print(f"   Config: {args.config}")
    
    config = load_config(args.config)
    
    # Check runtime configuration
    runtime_cfg = config.get('runtime', {})
    infer_cfg = runtime_cfg.get('infer_realtime', {})
    
    graph_order = infer_cfg.get('graph', {}).get('order', [])
    print(f"   Inference graph: {len(graph_order)} stages")
    for i, stage in enumerate(graph_order):
        print(f"     {i+1}. {stage}")
    
    returns = infer_cfg.get('graph', {}).get('returns', [])
    print(f"   Returns: {returns}")
    
    print("   🌐 Server simulation started! (Press Ctrl+C to stop)")
    print("   Endpoints:")
    print("     POST /predict - Run inference on video clip")
    print("     GET /health - Health check")
    print("     GET /metrics - Model metrics")
    
    try:
        import time
        time.sleep(2)  # Simulate server running
        print("   Server simulation completed!")
    except KeyboardInterrupt:
        print("   Server stopped by user")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TRIDENT-Net: Multimodal Fusion System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', default='tasks.yml', help='Configuration file')
    train_parser.add_argument('--jsonl', default=None, help='Override data.sources.jsonl_path')
    train_parser.add_argument('--video-root', dest='video_root', default=None, help='Override data.sources.video_root')
    train_parser.add_argument(
        '--pipeline',
        choices=['normal', 'finaltrain'],
        default='normal',
        help='Training pipeline'
    )
    train_parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    train_parser.add_argument('--finaltrain', action='store_true', help='Final training stage')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate models')
    eval_parser.add_argument('--config', default='tasks.yml', help='Configuration file')
    eval_parser.add_argument('--jsonl', default=None, help='Override data.sources.jsonl_path')
    eval_parser.add_argument('--video-root', dest='video_root', default=None, help='Override data.sources.video_root')
    eval_parser.add_argument('--report', default='./runs/eval_report.json', help='Report output path')
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--config', default='tasks.yml', help='Configuration file')
    tune_parser.add_argument('--component', default='f2', help='Component to tune')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start inference server')
    serve_parser.add_argument('--config', default='tasks.yml', help='Configuration file')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == 'train':
        command_train(args)
    elif args.command == 'eval':
        command_eval(args)
    elif args.command == 'tune':
        command_tune(args)
    elif args.command == 'serve':
        command_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()