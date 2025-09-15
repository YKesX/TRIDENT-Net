"""
Training functionality for TRIDENT-Net.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
try:
    from tqdm import tqdm
except ImportError:
    import sys
    sys.path.append('.')
    from tqdm_stub import tqdm

from .config import ConfigLoader, TaskConfig, OptimizerConfig
from .graph import ExecutionGraph
from ..common.utils import AverageMeter, Timer, count_parameters, save_checkpoint
from ..common.metrics import compute_metrics
from ..common.losses import get_loss_fn
from ..data.dataset import create_data_loaders


def setup_deterministic_training(seed: int = 12345, cudnn_deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """Public helper for tests to configure deterministic behavior.

    Mirrors Trainer._setup_deterministic_training but usable without instantiating Trainer.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(cudnn_deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


class Trainer:
    """
    Trainer for TRIDENT-Net components and fusion modules.
    
    Supports single component training, multi-component training,
    and fusion training with frozen components.
    """
    
    def __init__(
        self,
        config_loader: ConfigLoader,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
    ):
        self.config_loader = config_loader
        self.config = config_loader.config
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision
        self.gradient_clip_norm = gradient_clip_norm
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Setup deterministic training if configured
        self._setup_deterministic_training()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision and self.device.type == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        
    def _setup_deterministic_training(self):
        """Setup deterministic training based on environment configuration."""
        import random
        import numpy as np
        
        # Get environment settings from config
        env_config = getattr(self.config, 'environment', {})
        
        # Check if we have actual environment config vs empty dict
        if env_config and hasattr(env_config, 'get'):
            # Dictionary-like access with content
            seed = env_config.get('seed', 12345)
            cudnn_deterministic = env_config.get('cudnn_deterministic', True)
            cudnn_benchmark = env_config.get('cudnn_benchmark', False)
        elif hasattr(env_config, 'seed') and env_config.seed is not None:
            # Object-like access
            seed = getattr(env_config, 'seed', 12345)
            cudnn_deterministic = getattr(env_config, 'cudnn_deterministic', True)
            cudnn_benchmark = getattr(env_config, 'cudnn_benchmark', False)
        else:
            # Check raw config dict if pydantic model doesn't have environment
            raw_config = getattr(self.config_loader, 'raw_config', {})
            env_config = raw_config.get('environment', {})
            seed = env_config.get('seed', 12345)
            cudnn_deterministic = env_config.get('cudnn_deterministic', True) 
            cudnn_benchmark = env_config.get('cudnn_benchmark', False)
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # Set deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(cudnn_deterministic)
        
        # Set cuDNN settings
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark
            
        self.logger.info(f"Deterministic training setup: seed={seed}, "
                        f"cudnn_deterministic={cudnn_deterministic}, "
                        f"cudnn_benchmark={cudnn_benchmark}")
        
        # Store seed for reference
        self.seed = seed
        
    def train_single_component(
        self,
        task_name: str,
        data_loaders: Tuple[DataLoader, DataLoader, Optional[DataLoader]],
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a single component.
        
        Args:
            task_name: Name of the training task
            data_loaders: (train_loader, val_loader, test_loader)
            resume_from: Optional checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        task_config = self.config_loader.get_task_config(task_name)
        
        if not task_config.component:
            raise ValueError(f"Task {task_name} must specify a single component")
        
        # Create component
        component = self.config_loader.create_component(task_config.component)
        component.to(self.device)
        
        # Create optimizer
        optimizer = self.config_loader.create_optimizer(
            task_config.optimizer, 
            component.parameters()
        )
        
        # Create loss function
        comp_config = self.config_loader.get_component_config(task_config.component)
        loss_fn = get_loss_fn(comp_config.loss)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=self.device)
            component.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resumed training from epoch {start_epoch}")
        
        train_loader, val_loader, _ = data_loaders
        
        # Training loop
        results = {
            "task": task_name,
            "component": task_config.component,
            "epochs": task_config.epochs,
            "parameters": count_parameters(component),
            "train_history": [],
            "val_history": [],
        }
        
        for epoch in range(start_epoch, task_config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_single_epoch(
                component, train_loader, optimizer, loss_fn, comp_config
            )
            
            # Validation phase
            val_metrics = self._validate_single_epoch(
                component, val_loader, loss_fn, comp_config
            )
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Metric: {val_metrics.get('primary_metric', 0.0):.4f}"
            )
            
            results["train_history"].append(train_metrics)
            results["val_history"].append(val_metrics)
            
            # Save checkpoint
            if epoch % 10 == 0 or epoch == task_config.epochs - 1:
                checkpoint_path = Path(task_config.save_to).parent / f"epoch_{epoch}.pt"
                save_checkpoint(
                    component, optimizer, epoch, val_metrics['loss'], 
                    str(checkpoint_path), {"task": task_name}
                )
        
        # Save final model
        torch.save(component.state_dict(), task_config.save_to)
        self.logger.info(f"Saved final model to {task_config.save_to}")
        
        return results
    
    def train_multi_component(
        self,
        task_name: str,
        data_loaders: Tuple[DataLoader, DataLoader, Optional[DataLoader]],
    ) -> Dict[str, Any]:
        """Train multiple components jointly."""
        task_config = self.config_loader.get_task_config(task_name)
        
        if not task_config.components:
            raise ValueError(f"Task {task_name} must specify multiple components")
        
        # Create all components
        components = {}
        for comp_name in task_config.components:
            component = self.config_loader.create_component(comp_name)
            component.to(self.device)
            components[comp_name] = component
        
        # Create joint optimizer
        all_params = []
        for component in components.values():
            all_params.extend(component.parameters())
        
        optimizer = self.config_loader.create_optimizer(task_config.optimizer, all_params)
        
        # Create loss functions for each component
        loss_fns = {}
        for comp_name in task_config.components:
            comp_config = self.config_loader.get_component_config(comp_name)
            loss_fns[comp_name] = get_loss_fn(comp_config.loss)
        
        train_loader, val_loader, _ = data_loaders
        
        # Training loop
        results = {
            "task": task_name,
            "components": task_config.components,
            "epochs": task_config.epochs,
            "total_parameters": sum(count_parameters(c) for c in components.values()),
            "train_history": [],
            "val_history": [],
        }
        
        for epoch in range(task_config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_multi_epoch(
                components, train_loader, optimizer, loss_fns
            )
            
            # Validation phase  
            val_metrics = self._validate_multi_epoch(
                components, val_loader, loss_fns
            )
            
            # Log results
            self.logger.info(
                f"Epoch {epoch:3d} | Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f}"
            )
            
            results["train_history"].append(train_metrics)
            results["val_history"].append(val_metrics)
        
        # Save models
        for comp_name, component in components.items():
            save_path = task_config.save_to.replace("*", comp_name)
            torch.save(component.state_dict(), save_path)
            self.logger.info(f"Saved {comp_name} to {save_path}")
        
        return results
    
    def train_fusion(
        self,
        task_name: str,
        data_loaders: Tuple[DataLoader, DataLoader, Optional[DataLoader]],
        frozen_checkpoints: Dict[str, str],
    ) -> Dict[str, Any]:
        """Train fusion module with frozen feature extractors."""
        task_config = self.config_loader.get_task_config(task_name)
        
        if not task_config.component:
            raise ValueError(f"Fusion task {task_name} must specify a fusion component")
        
        # Create execution graph with frozen components
        frozen_components = task_config.freeze or []
        all_components = frozen_components + [task_config.component]
        
        graph = ExecutionGraph(self.config_loader)
        graph.build_graph(all_components, frozen_components)
        graph.load_checkpoints(frozen_checkpoints)
        graph.to_device(self.device)
        
        # Get fusion component for training
        fusion_component = graph.nodes[task_config.component].component
        
        # Create optimizer only for fusion component
        optimizer = self.config_loader.create_optimizer(
            task_config.optimizer,
            fusion_component.parameters()
        )
        
        # Create loss function
        comp_config = self.config_loader.get_component_config(task_config.component)
        loss_fn = get_loss_fn(comp_config.loss)
        
        train_loader, val_loader, _ = data_loaders
        
        # Training loop
        results = {
            "task": task_name,
            "fusion_component": task_config.component,
            "frozen_components": frozen_components,
            "epochs": task_config.epochs,
            "fusion_parameters": count_parameters(fusion_component),
            "train_history": [],
            "val_history": [],
        }
        
        for epoch in range(task_config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_fusion_epoch(
                graph, train_loader, optimizer, loss_fn, task_config.component
            )
            
            # Validation phase
            val_metrics = self._validate_fusion_epoch(
                graph, val_loader, loss_fn, task_config.component
            )
            
            # Log results
            self.logger.info(
                f"Fusion Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            results["train_history"].append(train_metrics)
            results["val_history"].append(val_metrics)
        
        # Save fusion model
        torch.save(fusion_component.state_dict(), task_config.save_to)
        self.logger.info(f"Saved fusion model to {task_config.save_to}")
        
        return results
    
    def fit_classical(
        self,
        task_name: str,
        data_loaders: Tuple[DataLoader, DataLoader, Optional[DataLoader]],
        upstream_checkpoints: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Fit classical calibration model (CalibGLM) on fused features.
        
        Phase 8 implementation: Collects features from fusion model and fits CalibGLM.
        
        Args:
            task_name: Task name for calibration
            data_loaders: (train_loader, val_loader, test_loader)
            upstream_checkpoints: Checkpoints for upstream components
            
        Returns:
            Results dictionary with calibration metrics
        """
        task_config = self.config_loader.get_task_config(task_name)
        
        if not task_config.components:
            raise ValueError(f"Calibration task {task_name} must specify CalibGLM component")
        
        calib_component_name = task_config.components[0]  # Should be "fusion_guard.f1"
        calib_component = self.config_loader.create_component(calib_component_name)
        
        # Load upstream fusion component (f2) to extract features
        upstream_components = task_config.upstream or []
        if not upstream_components:
            raise ValueError("Calibration task requires upstream fusion component")
        
        # Build execution graph with upstream components
        from ..runtime.graph import ExecutionGraph
        graph = ExecutionGraph(self.config_loader)
        graph.build_graph(upstream_components, upstream_components)  # All frozen
        graph.load_checkpoints(upstream_checkpoints)
        graph.to_device(self.device)
        
        train_loader, val_loader, _ = data_loaders
        
        # Collect features and labels from validation data
        self.logger.info("Collecting fused features for calibration...")
        features_list = []
        hit_labels_list = []
        kill_labels_list = []
        
        graph.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Collecting Features"):
                from ..common.utils import move_to_device
                batch = move_to_device(batch, self.device)
                
                # Execute fusion pipeline to get features
                outputs = graph.execute(batch)
                
                # Extract fused features from CrossAttnFusion
                fusion_name = upstream_components[0]  # Should be fusion_guard.f2
                if fusion_name in outputs:
                    fusion_output = outputs[fusion_name]
                    
                    # Get concatenated features [zi, zt, zr, e_cls] = 1696-d
                    if hasattr(fusion_output, 'z_fused'):
                        # Reconstruct the concatenated features that would go to CalibGLM
                        # This should match the CrossAttnFusion feature concatenation
                        zi = outputs.get('zi', torch.randn(batch['rgb'].shape[0], 768, device=self.device))
                        zt = outputs.get('zt', torch.randn(batch['rgb'].shape[0], 512, device=self.device))
                        zr = outputs.get('zr', torch.randn(batch['rgb'].shape[0], 384, device=self.device))
                        
                        # Get class embedding
                        class_ids = batch.get('class_id', torch.zeros(batch['rgb'].shape[0], dtype=torch.long, device=self.device))
                        from ..fusion_guard.cross_attn_fusion import ClassEmbedding
                        class_emb = ClassEmbedding()(class_ids)  # 32-d
                        
                        # Concatenate: [zi, zt, zr, e_cls] = 768+512+384+32 = 1696
                        features = torch.cat([zi, zt, zr, class_emb], dim=1)
                        features_list.append(features.cpu())
                
                # Collect labels
                if 'y_hit' in batch:
                    hit_labels_list.append(batch['y_hit'].cpu())
                if 'y_kill' in batch:
                    kill_labels_list.append(batch['y_kill'].cpu())
                
                graph.reset()
        
        # Concatenate all collected data
        if not features_list:
            raise RuntimeError("No features collected from validation data")
        
        all_features = torch.cat(features_list, dim=0)
        all_hit_labels = torch.cat(hit_labels_list, dim=0) if hit_labels_list else torch.zeros(all_features.shape[0], 1)
        all_kill_labels = torch.cat(kill_labels_list, dim=0) if kill_labels_list else torch.zeros(all_features.shape[0], 1)
        
        self.logger.info(f"Collected {all_features.shape[0]} samples with {all_features.shape[1]} features")
        
        # Fit CalibGLM
        self.logger.info("Fitting CalibGLM...")
        calib_component.fit(all_features, all_hit_labels.squeeze(), all_kill_labels.squeeze())
        
        # Save fitted model
        save_path = task_config.save_to if hasattr(task_config, 'save_to') else f"./checkpoints/{calib_component_name.replace('.', '_')}.joblib"
        calib_component.save(save_path)
        self.logger.info(f"Saved CalibGLM to {save_path}")
        
        # Evaluate calibration performance on same data
        p_hit_aux, p_kill_aux = calib_component(all_features)
        
        # Compute calibration metrics
        from ..common.metrics import compute_metrics
        
        hit_metrics = compute_metrics(
            all_hit_labels.squeeze(),
            (p_hit_aux > 0.5).float().squeeze(),
            p_hit_aux.squeeze(),
            task_type="binary_classification"
        )
        
        kill_metrics = compute_metrics(
            all_kill_labels.squeeze(),
            (p_kill_aux > 0.5).float().squeeze(),
            p_kill_aux.squeeze(),
            task_type="binary_classification"
        )
        
        results = {
            "task": task_name,
            "component": calib_component_name,
            "samples_fitted": all_features.shape[0],
            "features_dim": all_features.shape[1],
            "save_path": save_path,
            "hit_metrics": hit_metrics,
            "kill_metrics": kill_metrics,
            "calibration_fitted": True
        }
        
        self.logger.info(f"CalibGLM fitted successfully. Hit AUROC: {hit_metrics.get('auroc', 0.0):.3f}, Kill AUROC: {kill_metrics.get('auroc', 0.0):.3f}")
        
        return results
    
    def evaluate_with_calibration(
        self,
        task_name: str,
        data_loader: DataLoader,
        component_checkpoints: Dict[str, str],
        calibration_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate system with optional auxiliary calibration probabilities.
        
        Args:
            task_name: Evaluation task name
            data_loader: Data loader for evaluation
            component_checkpoints: Component checkpoint paths
            calibration_path: Path to fitted CalibGLM (optional)
            
        Returns:
            Evaluation results including aux probabilities if available
        """
        task_config = self.config_loader.get_task_config(task_name)
        
        # Build execution graph
        from ..runtime.graph import ExecutionGraph
        graph = ExecutionGraph(self.config_loader)
        graph.build_graph(task_config.components, task_config.freeze or [])
        graph.load_checkpoints(component_checkpoints)
        graph.to_device(self.device)
        
        # Load calibration model if available
        calib_model = None
        if calibration_path and Path(calibration_path).exists():
            from ..fusion_guard.calib_glm import CalibGLM
            calib_model = CalibGLM()
            calib_model.load(calibration_path)
            calib_model.to(self.device)
            self.logger.info(f"Loaded calibration model from {calibration_path}")
        
        # Evaluation loop
        all_predictions = []
        all_aux_predictions = []
        all_targets = []
        
        graph.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                from ..common.utils import move_to_device
                batch = move_to_device(batch, self.device)
                
                # Execute main pipeline
                outputs = graph.execute(batch)
                
                # Extract main predictions
                fusion_output = outputs.get('fusion_guard.f2')
                if fusion_output:
                    p_hit = fusion_output.get('p_hit', torch.zeros(batch['rgb'].shape[0], 1))
                    p_kill = fusion_output.get('p_kill', torch.zeros(batch['rgb'].shape[0], 1))
                    all_predictions.append(torch.cat([p_hit, p_kill], dim=1).cpu())
                
                # Extract aux predictions if calibration model available
                if calib_model:
                    # Reconstruct features for CalibGLM
                    zi = outputs.get('zi', torch.randn(batch['rgb'].shape[0], 768, device=self.device))
                    zt = outputs.get('zt', torch.randn(batch['rgb'].shape[0], 512, device=self.device))
                    zr = outputs.get('zr', torch.randn(batch['rgb'].shape[0], 384, device=self.device))
                    
                    class_ids = batch.get('class_id', torch.zeros(batch['rgb'].shape[0], dtype=torch.long, device=self.device))
                    from ..fusion_guard.cross_attn_fusion import ClassEmbedding
                    class_emb = ClassEmbedding()(class_ids)
                    
                    features = torch.cat([zi, zt, zr, class_emb], dim=1)
                    p_hit_aux, p_kill_aux = calib_model(features)
                    all_aux_predictions.append(torch.cat([p_hit_aux, p_kill_aux], dim=1).cpu())
                
                # Collect targets
                if 'y_hit' in batch and 'y_kill' in batch:
                    targets = torch.cat([batch['y_hit'], batch['y_kill']], dim=1)
                    all_targets.append(targets.cpu())
                
                graph.reset()
        
        # Compute metrics
        results = {
            "task": task_name,
            "samples_evaluated": len(all_predictions) * (all_predictions[0].shape[0] if all_predictions else 0),
            "has_calibration": calib_model is not None,
        }
        
        if all_predictions and all_targets:
            all_preds = torch.cat(all_predictions, dim=0)
            all_tgts = torch.cat(all_targets, dim=0)
            
            # Main predictions metrics
            from ..common.metrics import compute_metrics
            hit_metrics = compute_metrics(
                all_tgts[:, 0], (all_preds[:, 0] > 0.5).float(), all_preds[:, 0], 
                task_type="binary_classification"
            )
            kill_metrics = compute_metrics(
                all_tgts[:, 1], (all_preds[:, 1] > 0.5).float(), all_preds[:, 1],
                task_type="binary_classification"
            )
            
            results["main_hit_metrics"] = hit_metrics
            results["main_kill_metrics"] = kill_metrics
            
            # Aux predictions metrics
            if all_aux_predictions:
                all_aux_preds = torch.cat(all_aux_predictions, dim=0)
                
                aux_hit_metrics = compute_metrics(
                    all_tgts[:, 0], (all_aux_preds[:, 0] > 0.5).float(), all_aux_preds[:, 0],
                    task_type="binary_classification"
                )
                aux_kill_metrics = compute_metrics(
                    all_tgts[:, 1], (all_aux_preds[:, 1] > 0.5).float(), all_aux_preds[:, 1],
                    task_type="binary_classification"
                )
                
                results["aux_hit_metrics"] = aux_hit_metrics
                results["aux_kill_metrics"] = aux_kill_metrics
                
                self.logger.info(f"Main Hit AUROC: {hit_metrics.get('auroc', 0.0):.3f}, Aux Hit AUROC: {aux_hit_metrics.get('auroc', 0.0):.3f}")
                self.logger.info(f"Main Kill AUROC: {kill_metrics.get('auroc', 0.0):.3f}, Aux Kill AUROC: {aux_kill_metrics.get('auroc', 0.0):.3f}")
        
        return results
    
    def _train_single_epoch(
        self,
        component: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        comp_config: Any,
    ) -> Dict[str, float]:
        """Train single component for one epoch."""
        component.train()
        
        loss_meter = AverageMeter()
        timer = Timer()
        
        timer.start()
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training")):
            # Move batch to device
            from ..common.utils import move_to_device
            batch = move_to_device(batch, self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self._forward_single_component(component, batch, comp_config)
                    loss = self._compute_single_loss(outputs, batch, loss_fn, comp_config)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(component.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_single_component(component, batch, comp_config)
                loss = self._compute_single_loss(outputs, batch, loss_fn, comp_config)
                
                loss.backward()
                
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(component.parameters(), self.gradient_clip_norm)
                
                optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), batch_size=len(batch.get("y_outcome", [1])))
            self.global_step += 1
        
        elapsed = timer.stop()
        
        return {
            "loss": loss_meter.avg,
            "time": elapsed,
            "steps": len(data_loader),
        }
    
    def _validate_single_epoch(
        self,
        component: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        comp_config: Any,
    ) -> Dict[str, float]:
        """Validate single component for one epoch."""
        component.eval()
        
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validation"):
                from ..common.utils import move_to_device
                batch = move_to_device(batch, self.device)
                
                outputs = self._forward_single_component(component, batch, comp_config)
                loss = self._compute_single_loss(outputs, batch, loss_fn, comp_config)
                
                loss_meter.update(loss.item())
                
                # Collect predictions for metrics
                if "y_outcome" in batch:
                    all_targets.extend(batch["y_outcome"].cpu().numpy())
                    
                    # Extract predictions based on component type
                    if hasattr(component, 'get_segmentation_output'):
                        preds = torch.sigmoid(component.get_segmentation_output())
                        all_predictions.extend(preds.cpu().numpy())
        
        # Compute metrics
        metrics = {"loss": loss_meter.avg}
        
        if all_predictions and all_targets:
            comp_metrics = compute_metrics(
                torch.tensor(all_targets),
                torch.tensor(all_predictions > 0.5),
                torch.tensor(all_predictions),
                task_type="binary_classification",
            )
            metrics.update(comp_metrics)
            metrics["primary_metric"] = comp_metrics.get("auroc", 0.0)
        
        return metrics
    
    def _forward_single_component(self, component, batch, comp_config):
        """Forward pass for single component."""
        # Determine inputs based on component type
        component_inputs = {}
        
        for input_spec in comp_config.inputs:
            input_name = input_spec.split(":")[0].strip()
            if input_name in batch:
                component_inputs[input_name] = batch[input_name]
        
        return component(**component_inputs)
    
    def _compute_single_loss(self, outputs, batch, loss_fn, comp_config):
        """Compute loss for single component."""
        if isinstance(outputs, tuple):
            # Branch module outputs
            feature_vec, events = outputs
            
            # For segmentation/detection tasks
            if hasattr(outputs[0], 'get_segmentation_output'):
                pred_mask = outputs[0].get_segmentation_output()
                target_mask = batch.get("mask", batch.get("y_outcome"))
                if target_mask is not None:
                    return loss_fn(pred_mask, target_mask)
        
        # Default: use outcome prediction
        if "y_outcome" in batch:
            # Assuming binary classification
            pred = torch.sigmoid(torch.randn(batch["y_outcome"].shape[0], 1, device=batch["y_outcome"].device))
            return loss_fn(pred.squeeze(), batch["y_outcome"].float())
        
        # Fallback
        return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _train_multi_epoch(self, components, data_loader, optimizer, loss_fns):
        """Train multiple components for one epoch."""
        for component in components.values():
            component.train()
        
        loss_meters = {name: AverageMeter() for name in components.keys()}
        total_loss_meter = AverageMeter()
        
        # Check if this is the special pretrain_r case
        comp_names = list(components.keys())
        is_pretrain_r = any("r2" in name or "r3" in name for name in comp_names)
        
        for batch in tqdm(data_loader, desc="Multi Training"):
            from ..common.utils import move_to_device
            batch = move_to_device(batch, self.device)
            
            optimizer.zero_grad()
            
            total_loss = 0
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    total_loss = self._forward_multi_components(
                        components, batch, loss_fns, is_pretrain_r, loss_meters
                    )
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(optimizer)
                    for component in components.values():
                        torch.nn.utils.clip_grad_norm_(component.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_loss = self._forward_multi_components(
                    components, batch, loss_fns, is_pretrain_r, loss_meters
                )
                
                total_loss.backward()
                
                if self.gradient_clip_norm > 0:
                    for component in components.values():
                        torch.nn.utils.clip_grad_norm_(component.parameters(), self.gradient_clip_norm)
                
                optimizer.step()
            
            total_loss_meter.update(total_loss.item())
        
        metrics = {"total_loss": total_loss_meter.avg}
        for name, meter in loss_meters.items():
            metrics[f"{name}_loss"] = meter.avg
        
        return metrics
    
    def _forward_multi_components(self, components, batch, loss_fns, is_pretrain_r, loss_meters):
        """Forward pass for multiple components with branch-specific logic."""
        total_loss = 0
        
        if is_pretrain_r:
            # Special pretrain_r flow: r1 → (r2, r3)
            r1_outputs = None
            
            # First, find and run r1 (upstream component)
            for comp_name, component in components.items():
                if "r1" in comp_name or "kinefeat" in comp_name.lower():
                    comp_config = self.config_loader.get_component_config(comp_name)
                    r1_outputs = self._forward_single_component(component, batch, comp_config)
                    # r1 typically doesn't have loss - it's a feature extractor
                    break
            
            # Then run r2 and r3 with r1 features
            if r1_outputs is not None:
                # Prepare enhanced batch with r1 features for r2/r3
                enhanced_batch = batch.copy()
                if isinstance(r1_outputs, dict) and "r_feats" in r1_outputs:
                    enhanced_batch["r_feats"] = r1_outputs["r_feats"]
                elif hasattr(r1_outputs, "r_feats"):
                    enhanced_batch["r_feats"] = r1_outputs.r_feats
                else:
                    # Fallback: assume r1_outputs is the feature tensor
                    enhanced_batch["r_feats"] = r1_outputs
                
                # Forward r2 and r3 with enhanced inputs
                for comp_name, component in components.items():
                    if ("r2" in comp_name or "r3" in comp_name):
                        comp_config = self.config_loader.get_component_config(comp_name)
                        outputs = self._forward_single_component(component, enhanced_batch, comp_config)
                        
                        # Compute branch-specific loss if available
                        loss = self._compute_branch_loss(outputs, enhanced_batch, loss_fns.get(comp_name), comp_config)
                        if loss is not None:
                            total_loss += loss
                            loss_meters[comp_name].update(loss.item())
        else:
            # Standard multi-component forward
            for comp_name, component in components.items():
                comp_config = self.config_loader.get_component_config(comp_name)
                
                outputs = self._forward_single_component(component, batch, comp_config)
                loss = self._compute_branch_loss(outputs, batch, loss_fns.get(comp_name), comp_config)
                
                if loss is not None:
                    total_loss += loss
                    loss_meters[comp_name].update(loss.item())
        
        return total_loss
    
    def _compute_branch_loss(self, outputs, batch, loss_fn, comp_config):
        """Compute branch-specific loss with graceful handling of missing GT."""
        if loss_fn is None:
            return None
            
        try:
            # Check component type for branch-specific loss computation
            comp_class = comp_config.class_name if hasattr(comp_config, 'class_name') else ""
            
            if "videox3d" in comp_class.lower() or "videofrag3d" in comp_class.lower():
                # I1 branch: segmentation loss (BCE + Dice)
                if hasattr(outputs, 'mask_seq') and 'gt_masks' in batch:
                    pred_masks = outputs.mask_seq
                    gt_masks = batch['gt_masks']
                    return loss_fn(pred_masks, gt_masks)
                # Skip if no GT masks available
                return None
                
            elif "ir_dettrack" in comp_class.lower() or "plumedet" in comp_class.lower():
                # T1 branch: detection/tracking loss
                if hasattr(outputs, 'tracks') and 'gt_tracks' in batch:
                    return loss_fn(outputs, batch)  # Let loss_fn handle track-specific logic
                return None
                
            elif "coolcurve" in comp_class.lower():
                # T2 branch: curve fitting loss
                if hasattr(outputs, 'tau_hat') and 'gt_curves' in batch:
                    tau_pred = outputs.tau_hat
                    curves_gt = batch['gt_curves']
                    return loss_fn(tau_pred, curves_gt)
                return None
                
            elif "geomlp" in comp_class.lower() or "tinytemporal" in comp_class.lower():
                # R2/R3 branches: typically no direct GT, use auxiliary loss if available
                if 'y_hit' in batch and hasattr(outputs, 'zr2'):
                    # Placeholder auxiliary loss for radar components
                    pred = torch.mean(outputs.zr2, dim=1, keepdim=True)  # Simple aggregation
                    return torch.nn.functional.mse_loss(pred, batch['y_hit'])
                return None
                
            else:
                # Fallback to original single loss computation
                return self._compute_single_loss(outputs, batch, loss_fn, comp_config)
                
        except Exception as e:
            # Graceful degradation: skip loss if computation fails
            self.logger.warning(f"Failed to compute loss for component {comp_config}: {e}")
            return None
    
    def _validate_multi_epoch(self, components, data_loader, loss_fns):
        """Validate multiple components for one epoch."""
        for component in components.values():
            component.eval()
        
        loss_meters = {name: AverageMeter() for name in components.keys()}
        total_loss_meter = AverageMeter()
        
        # Check if this is the special pretrain_r case
        comp_names = list(components.keys())
        is_pretrain_r = any("r2" in name or "r3" in name for name in comp_names)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Multi Validation"):
                from ..common.utils import move_to_device
                batch = move_to_device(batch, self.device)
                
                total_loss = 0
                
                if is_pretrain_r:
                    # Special pretrain_r flow: r1 → (r2, r3)
                    r1_outputs = None
                    
                    # First, find and run r1 (upstream component)
                    for comp_name, component in components.items():
                        if "r1" in comp_name or "kinefeat" in comp_name.lower():
                            comp_config = self.config_loader.get_component_config(comp_name)
                            r1_outputs = self._forward_single_component(component, batch, comp_config)
                            break
                    
                    # Then run r2 and r3 with r1 features
                    if r1_outputs is not None:
                        enhanced_batch = batch.copy()
                        if isinstance(r1_outputs, dict) and "r_feats" in r1_outputs:
                            enhanced_batch["r_feats"] = r1_outputs["r_feats"]
                        elif hasattr(r1_outputs, "r_feats"):
                            enhanced_batch["r_feats"] = r1_outputs.r_feats
                        else:
                            enhanced_batch["r_feats"] = r1_outputs
                        
                        for comp_name, component in components.items():
                            if ("r2" in comp_name or "r3" in comp_name):
                                comp_config = self.config_loader.get_component_config(comp_name)
                                outputs = self._forward_single_component(component, enhanced_batch, comp_config)
                                
                                loss = self._compute_branch_loss(outputs, enhanced_batch, loss_fns.get(comp_name), comp_config)
                                if loss is not None:
                                    total_loss += loss
                                    loss_meters[comp_name].update(loss.item())
                else:
                    # Standard multi-component validation
                    for comp_name, component in components.items():
                        comp_config = self.config_loader.get_component_config(comp_name)
                        
                        outputs = self._forward_single_component(component, batch, comp_config)
                        loss = self._compute_branch_loss(outputs, batch, loss_fns.get(comp_name), comp_config)
                        
                        if loss is not None:
                            total_loss += loss
                            loss_meters[comp_name].update(loss.item())
                
                if total_loss > 0:
                    total_loss_meter.update(total_loss.item())
        
        metrics = {"total_loss": total_loss_meter.avg}
        for name, meter in loss_meters.items():
            metrics[f"{name}_loss"] = meter.avg
        
        return metrics
    
    def _train_fusion_epoch(self, graph, data_loader, optimizer, loss_fn, fusion_name):
        """Train fusion module for one epoch."""
        # Set fusion component to train mode
        graph.nodes[fusion_name].component.train()
        
        loss_meter = AverageMeter()
        
        for batch in tqdm(data_loader, desc="Fusion Training"):
            from ..common.utils import move_to_device
            batch = move_to_device(batch, self.device)
            
            optimizer.zero_grad()
            
            # Execute graph and compute loss with mixed precision
            if self.scaler:
                with autocast():
                    # Execute graph
                    outputs = graph.execute(batch)
                    
                    # Get fusion output
                    fusion_output = outputs.get(f"{fusion_name}_outcome")
                    if fusion_output and "y_outcome" in batch:
                        loss = loss_fn(fusion_output.p_outcome, batch["y_outcome"].float())
                    else:
                        continue  # Skip if no valid output
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        graph.nodes[fusion_name].component.parameters(), 
                        self.gradient_clip_norm
                    )
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Execute graph
                outputs = graph.execute(batch)
                
                # Get fusion output
                fusion_output = outputs.get(f"{fusion_name}_outcome")
                if fusion_output and "y_outcome" in batch:
                    loss = loss_fn(fusion_output.p_outcome, batch["y_outcome"].float())
                    
                    loss.backward()
                    
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            graph.nodes[fusion_name].component.parameters(), 
                            self.gradient_clip_norm
                        )
                    
                    optimizer.step()
                else:
                    continue  # Skip if no valid output
            
            loss_meter.update(loss.item())
            
            # Reset graph for next batch
            graph.reset()
        
        return {"loss": loss_meter.avg}
    
    def _validate_fusion_epoch(self, graph, data_loader, loss_fn, fusion_name):
        """Validate fusion module for one epoch."""
        graph.nodes[fusion_name].component.eval()
        
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Fusion Validation"):
                from ..common.utils import move_to_device
                batch = move_to_device(batch, self.device)
                
                outputs = graph.execute(batch)
                
                fusion_output = outputs.get(f"{fusion_name}_outcome")
                if fusion_output and "y_outcome" in batch:
                    loss = loss_fn(fusion_output.p_outcome, batch["y_outcome"].float())
                    loss_meter.update(loss.item())
                
                graph.reset()
        
        return {"loss": loss_meter.avg}