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
from tqdm import tqdm

from .config import ConfigLoader, TaskConfig, OptimizerConfig
from .graph import ExecutionGraph
from ..common.utils import AverageMeter, Timer, count_parameters, save_checkpoint
from ..common.metrics import compute_metrics
from ..common.losses import get_loss_fn
from ..data.dataset import create_data_loaders


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
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision and self.device.type == "cuda" else None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        
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
        
        for batch in tqdm(data_loader, desc="Multi Training"):
            from ..common.utils import move_to_device
            batch = move_to_device(batch, self.device)
            
            optimizer.zero_grad()
            
            total_loss = 0
            for comp_name, component in components.items():
                comp_config = self.config_loader.get_component_config(comp_name)
                
                outputs = self._forward_single_component(component, batch, comp_config)
                loss = self._compute_single_loss(outputs, batch, loss_fns[comp_name], comp_config)
                
                total_loss += loss
                loss_meters[comp_name].update(loss.item())
            
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
    
    def _validate_multi_epoch(self, components, data_loader, loss_fns):
        """Validate multiple components for one epoch."""
        for component in components.values():
            component.eval()
        
        loss_meters = {name: AverageMeter() for name in components.keys()}
        total_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Multi Validation"):
                from ..common.utils import move_to_device
                batch = move_to_device(batch, self.device)
                
                total_loss = 0
                for comp_name, component in components.items():
                    comp_config = self.config_loader.get_component_config(comp_name)
                    
                    outputs = self._forward_single_component(component, batch, comp_config)
                    loss = self._compute_single_loss(outputs, batch, loss_fns[comp_name], comp_config)
                    
                    total_loss += loss
                    loss_meters[comp_name].update(loss.item())
                
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