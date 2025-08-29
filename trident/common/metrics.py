"""
Evaluation metrics for TRIDENT-Net.

Author: Yağızhan Keskin  
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def auroc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Area Under ROC Curve."""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_score_np = y_score.detach().cpu().numpy().flatten()
    
    if len(np.unique(y_true_np)) < 2:
        return float('nan')
    
    return roc_auc_score(y_true_np, y_score_np)


def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Average Precision Score."""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_score_np = y_score.detach().cpu().numpy().flatten()
    
    if len(np.unique(y_true_np)) < 2:
        return float('nan')
    
    return average_precision_score(y_true_np, y_score_np)


def f1(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> float:
    """F1 Score with optional threshold."""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    
    if y_pred.dtype == torch.bool or torch.all((y_pred == 0) | (y_pred == 1)):
        y_pred_np = y_pred.detach().cpu().numpy().flatten()
    else:
        y_pred_np = (y_pred.detach().cpu().numpy().flatten() > threshold).astype(int)
    
    return f1_score(y_true_np, y_pred_np, zero_division=0.0)


def brier_score(y_true: torch.Tensor, y_prob: torch.Tensor) -> float:
    """Brier Score for probability calibration."""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_prob_np = y_prob.detach().cpu().numpy().flatten()
    
    return np.mean((y_prob_np - y_true_np) ** 2)


def expected_calibration_error(
    y_true: torch.Tensor, 
    y_prob: torch.Tensor, 
    n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE)."""
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_prob_np = y_prob.detach().cpu().numpy().flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob_np > bin_lower) & (y_prob_np <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true_np[in_bin].mean()
            avg_confidence_in_bin = y_prob_np[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def time_to_confidence(
    y_prob_sequence: torch.Tensor,
    threshold: float = 0.9,
    dt: float = 1.0,
) -> List[float]:
    """
    Time to reach confidence threshold for sequential predictions.
    
    Args:
        y_prob_sequence: Shape (B, T) - probabilities over time
        threshold: Confidence threshold
        dt: Time step in seconds
        
    Returns:
        List of times to reach threshold for each sample
    """
    times = []
    B, T = y_prob_sequence.shape
    
    for b in range(B):
        probs = y_prob_sequence[b].detach().cpu().numpy()
        time_steps = np.where(probs >= threshold)[0]
        
        if len(time_steps) > 0:
            times.append(time_steps[0] * dt)
        else:
            times.append(float('inf'))  # Never reached threshold
    
    return times


def mean_intersection_over_union(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
) -> float:
    """Mean IoU for segmentation tasks."""
    if y_pred.shape[1] > 1:  # Multi-class logits
        y_pred = torch.argmax(y_pred, dim=1)
    else:  # Binary
        y_pred = (y_pred > 0.5).long()
    
    y_true = y_true.long()
    
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_inds = (y_pred == cls)
        target_inds = (y_true == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return F.l1_loss(y_pred, y_true).item()


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: torch.Tensor,
    task_type: str = "binary_classification",
) -> dict:
    """
    Compute relevant metrics based on task type.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities
        task_type: Type of task (binary_classification, segmentation, regression)
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    if task_type == "binary_classification":
        metrics["auroc"] = auroc(y_true, y_prob)
        metrics["f1"] = f1(y_true, y_pred)
        metrics["brier"] = brier_score(y_true, y_prob)
        metrics["ece"] = expected_calibration_error(y_true, y_prob)
        metrics["ap"] = average_precision(y_true, y_prob)
        
    elif task_type == "segmentation":
        metrics["miou"] = mean_intersection_over_union(y_true, y_prob)
        metrics["f1"] = f1(y_true.flatten(), (y_prob > 0.5).flatten())
        
    elif task_type == "regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        
    return metrics