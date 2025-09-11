"""
Evaluation functionality for TRIDENT-Net.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    import sys
    sys.path.append('.')
    from tqdm_stub import tqdm
import numpy as np

from .config import ConfigLoader
from .graph import ExecutionGraph, create_inference_graph
from ..common.metrics import (
    auroc, f1, brier_score, expected_calibration_error, 
    time_to_confidence, compute_metrics
)
from ..common.utils import move_to_device


class Evaluator:
    """
    Evaluator for TRIDENT-Net system performance.
    
    Provides comprehensive evaluation of individual components,
    fusion modules, and overall system performance.
    """
    
    def __init__(
        self,
        config_loader: ConfigLoader,
        device: Optional[torch.device] = None,
    ):
        self.config_loader = config_loader
        self.config = config_loader.config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_system(
        self,
        components: List[str],
        test_loader: DataLoader,
        checkpoint_map: Dict[str, str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate complete TRIDENT system.
        
        Args:
            components: List of component names to evaluate
            test_loader: Test data loader
            checkpoint_map: Component checkpoint paths
            metrics: Metrics to compute
            
        Returns:
            Comprehensive evaluation results
        """
        if metrics is None:
            metrics = ["auroc", "f1", "brier", "ece", "time_to_0p9"]
        
        # Create inference graph
        graph = create_inference_graph(
            self.config,
            components,
            checkpoint_map,
            frozen_components=components,
            device=self.device,
        )
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_explanations = []
        component_outputs = {comp: [] for comp in components}
        
        graph.nodes[list(graph.nodes.keys())[0]].component.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                batch = move_to_device(batch, self.device)
                
                try:
                    # Execute graph
                    outputs = graph.execute(batch)
                    
                    # Collect final outcome
                    final_outcome = None
                    for key, value in outputs.items():
                        if "outcome" in key and hasattr(value, 'p_outcome'):
                            final_outcome = value
                            break
                    
                    if final_outcome and "y_outcome" in batch:
                        all_predictions.extend(final_outcome.p_outcome.cpu().numpy())
                        all_targets.extend(batch["y_outcome"].cpu().numpy())
                        
                        if final_outcome.explanation:
                            all_explanations.append(final_outcome.explanation)
                    
                    # Collect component-specific outputs
                    for comp_name in components:
                        comp_output = outputs.get(f"{comp_name}_outcome")
                        if comp_output:
                            component_outputs[comp_name].append(comp_output)
                    
                    # Reset graph
                    graph.reset()
                    
                except Exception as e:
                    self.logger.warning(f"Batch {batch_idx} failed: {e}")
                    continue
        
        # Compute overall metrics
        results = {
            "total_samples": len(all_targets),
            "components_evaluated": components,
        }
        
        if all_predictions and all_targets:
            pred_tensor = torch.tensor(all_predictions)
            target_tensor = torch.tensor(all_targets)
            binary_pred = (pred_tensor > 0.5).long()
            
            # Standard metrics
            if "auroc" in metrics:
                results["auroc"] = auroc(target_tensor, pred_tensor)
            
            if "f1" in metrics:
                results["f1"] = f1(target_tensor, binary_pred)
            
            if "brier" in metrics:
                results["brier"] = brier_score(target_tensor, pred_tensor)
            
            if "ece" in metrics:
                results["ece"] = expected_calibration_error(target_tensor, pred_tensor)
            
            # Time to confidence (if temporal data)
            if "time_to_0p9" in metrics and len(pred_tensor.shape) > 1:
                ttc = time_to_confidence(pred_tensor, threshold=0.9)
                results["time_to_confidence_0p9"] = np.mean(ttc)
        
        # Component-specific metrics
        for comp_name, outputs in component_outputs.items():
            if outputs:
                comp_results = self._evaluate_component_outputs(
                    comp_name, outputs, all_targets, metrics
                )
                results[f"{comp_name}_metrics"] = comp_results
        
        # Explanation analysis
        if all_explanations:
            explanation_stats = self._analyze_explanations(all_explanations)
            results["explanation_analysis"] = explanation_stats
        
        self.logger.info(f"Evaluation completed: {len(all_targets)} samples")
        
        return results
    
    def evaluate_single_component(
        self,
        component_name: str,
        test_loader: DataLoader,
        checkpoint_path: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single component in isolation."""
        if metrics is None:
            metrics = ["auroc", "f1", "brier"]
        
        # Load component
        component = self.config_loader.create_component(component_name)
        component.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            component.load_state_dict(checkpoint['model_state_dict'])
        else:
            component.load_state_dict(checkpoint)
        
        component.eval()
        
        # Collect outputs
        all_features = []
        all_events = []
        all_targets = []
        
        comp_config = self.config_loader.get_component_config(component_name)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {component_name}"):
                batch = move_to_device(batch, self.device)
                
                # Prepare component inputs
                comp_inputs = {}
                for input_spec in comp_config.inputs:
                    input_name = input_spec.split(":")[0].strip()
                    if input_name in batch:
                        comp_inputs[input_name] = batch[input_name]
                
                # Forward pass
                outputs = component(**comp_inputs)
                
                if isinstance(outputs, tuple):
                    feature_vec, events = outputs
                    all_features.append(feature_vec.z.cpu())
                    all_events.extend(events)
                
                if "y_outcome" in batch:
                    all_targets.extend(batch["y_outcome"].cpu().numpy())
        
        # Compute metrics
        results = {
            "component": component_name,
            "total_samples": len(all_targets),
            "total_events": len(all_events),
        }
        
        if all_features:
            # Feature statistics
            all_features_tensor = torch.cat(all_features, dim=0)
            results["feature_stats"] = {
                "mean_norm": torch.norm(all_features_tensor, dim=1).mean().item(),
                "std_norm": torch.norm(all_features_tensor, dim=1).std().item(),
                "feature_dim": all_features_tensor.shape[1],
            }
        
        # Event analysis
        if all_events:
            event_stats = self._analyze_events(all_events)
            results["event_analysis"] = event_stats
        
        return results
    
    def compare_fusion_methods(
        self,
        fusion_components: List[str],
        test_loader: DataLoader,
        checkpoint_maps: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """Compare different fusion methods."""
        comparison_results = {}
        
        for fusion_name in fusion_components:
            self.logger.info(f"Evaluating fusion method: {fusion_name}")
            
            # Get required components for this fusion method
            task_name = f"eval_{fusion_name}"
            
            try:
                # Create minimal task config for evaluation
                components = list(checkpoint_maps[fusion_name].keys())
                
                results = self.evaluate_system(
                    components,
                    test_loader,
                    checkpoint_maps[fusion_name],
                )
                
                comparison_results[fusion_name] = results
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {fusion_name}: {e}")
                comparison_results[fusion_name] = {"error": str(e)}
        
        # Create comparison summary
        summary = {
            "fusion_methods": list(comparison_results.keys()),
            "comparison_table": {},
        }
        
        metrics_to_compare = ["auroc", "f1", "brier", "ece"]
        
        for metric in metrics_to_compare:
            summary["comparison_table"][metric] = {}
            for fusion_name, results in comparison_results.items():
                if metric in results:
                    summary["comparison_table"][metric][fusion_name] = results[metric]
        
        comparison_results["summary"] = summary
        
        return comparison_results
    
    def _evaluate_component_outputs(
        self,
        component_name: str,
        outputs: List[Any],
        targets: List[float],
        metrics: List[str],
    ) -> Dict[str, float]:
        """Evaluate component-specific outputs."""
        component_metrics = {}
        
        # Extract predictions from component outputs
        predictions = []
        
        for output in outputs:
            if hasattr(output, 'p_outcome'):
                predictions.extend(output.p_outcome.cpu().numpy())
            elif hasattr(output, 'z'):
                # For feature vectors, use a simple linear classifier
                feature_norm = torch.norm(output.z, dim=1)
                prob = torch.sigmoid(feature_norm - feature_norm.mean())
                predictions.extend(prob.cpu().numpy())
        
        if predictions and len(predictions) == len(targets):
            pred_tensor = torch.tensor(predictions)
            target_tensor = torch.tensor(targets)
            
            # Compute requested metrics
            comp_metrics = compute_metrics(
                target_tensor,
                (pred_tensor > 0.5).long(),
                pred_tensor,
                task_type="binary_classification",
            )
            
            # Filter to requested metrics
            for metric in metrics:
                if metric in comp_metrics:
                    component_metrics[metric] = comp_metrics[metric]
        
        return component_metrics
    
    def _analyze_events(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze event statistics."""
        if not events:
            return {"total_events": 0}
        
        # Group by event type
        event_types = {}
        quality_scores = []
        durations = []
        
        for event in events:
            event_type = event.type
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
            
            quality_scores.append(event.quality)
            durations.append(event.t_end - event.t_start)
        
        return {
            "total_events": len(events),
            "unique_event_types": len(event_types),
            "event_type_counts": event_types,
            "avg_quality": np.mean(quality_scores),
            "avg_duration": np.mean(durations),
            "quality_std": np.std(quality_scores),
        }
    
    def _analyze_explanations(self, explanations: List[Dict]) -> Dict[str, Any]:
        """Analyze explanation statistics."""
        if not explanations:
            return {"total_explanations": 0}
        
        # Count explanation types
        explanation_types = {}
        fusion_types = {}
        
        for explanation in explanations:
            # Count fusion types
            fusion_type = explanation.get("fusion_type", "unknown")
            if fusion_type not in fusion_types:
                fusion_types[fusion_type] = 0
            fusion_types[fusion_type] += 1
            
            # Count explanation keys
            for key in explanation.keys():
                if key not in explanation_types:
                    explanation_types[key] = 0
                explanation_types[key] += 1
        
        return {
            "total_explanations": len(explanations),
            "fusion_type_distribution": fusion_types,
            "explanation_key_frequency": explanation_types,
        }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate formatted evaluation report."""
        report_lines = [
            "# TRIDENT-Net Evaluation Report",
            "",
            f"**Total Samples**: {results.get('total_samples', 0)}",
            f"**Components Evaluated**: {', '.join(results.get('components_evaluated', []))}",
            "",
            "## Overall Performance",
            "",
        ]
        
        # Overall metrics
        metrics_table = []
        for metric in ["auroc", "f1", "brier", "ece"]:
            if metric in results:
                metrics_table.append(f"| {metric.upper()} | {results[metric]:.4f} |")
        
        if metrics_table:
            report_lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
            ])
            report_lines.extend(metrics_table)
            report_lines.append("")
        
        # Component-specific results
        component_results = {k: v for k, v in results.items() if k.endswith("_metrics")}
        
        if component_results:
            report_lines.extend([
                "## Component Performance",
                "",
                "| Component | AUROC | F1 | Brier |",
                "|-----------|-------|----|----- --|",
            ])
            
            for comp_key, metrics in component_results.items():
                comp_name = comp_key.replace("_metrics", "")
                auroc_val = metrics.get("auroc", 0.0)
                f1_val = metrics.get("f1", 0.0)
                brier_val = metrics.get("brier", 0.0)
                
                report_lines.append(
                    f"| {comp_name} | {auroc_val:.4f} | {f1_val:.4f} | {brier_val:.4f} |"
                )
            
            report_lines.append("")
        
        # Event analysis
        if "explanation_analysis" in results:
            explanation_stats = results["explanation_analysis"]
            report_lines.extend([
                "## Explanation Analysis",
                "",
                f"- Total explanations: {explanation_stats.get('total_explanations', 0)}",
                f"- Fusion types: {list(explanation_stats.get('fusion_type_distribution', {}).keys())}",
                "",
            ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_text