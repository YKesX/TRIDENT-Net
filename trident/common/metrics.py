"""
Evaluation metrics for the TRIDENT system.

Implements standard classification and calibration metrics.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


def auroc(
    y_true: torch.Tensor, 
    y_scores: torch.Tensor
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probability scores
        
    Returns:
        AUROC score
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_scores_np = y_scores.detach().cpu().numpy()
    return roc_auc_score(y_true_np, y_scores_np)


def f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str = "binary"
) -> float:
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        average: Averaging method for multiclass
        
    Returns:
        F1 score
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return f1_score(y_true_np, y_pred_np, average=average)


def brier_score(
    y_true: torch.Tensor,
    y_probs: torch.Tensor
) -> float:
    """
    Compute Brier score for probability calibration.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        
    Returns:
        Brier score (lower is better)
    """
    return torch.mean((y_probs - y_true.float()) ** 2).item()


def expected_calibration_error(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        n_bins: Number of confidence bins
        
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].float().mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            
            # Add weighted difference to ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def time_to_confidence(
    y_probs_sequence: torch.Tensor,
    threshold: float = 0.8,
    dt: float = 1.0
) -> Optional[float]:
    """
    Compute time until confidence threshold is exceeded.
    
    Args:
        y_probs_sequence: Probability sequence over time (T,)
        threshold: Confidence threshold
        dt: Time step between frames
        
    Returns:
        Time to reach threshold, or None if never reached
    """
    exceeds_threshold = y_probs_sequence >= threshold
    if not exceeds_threshold.any():
        return None
    
    first_idx = torch.argmax(exceeds_threshold.float()).item()
    return first_idx * dt


def reliability_diagram_data(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram plotting.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        n_bins: Number of confidence bins
        
    Returns:
        Tuple of (bin_centers, accuracies, counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    y_true_np = y_true.detach().cpu().numpy()
    y_probs_np = y_probs.detach().cpu().numpy()
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_probs_np > bin_lower) & (y_probs_np <= bin_upper)
        
        if i == 0:  # Include left boundary for first bin
            in_bin = (y_probs_np >= bin_lower) & (y_probs_np <= bin_upper)
        
        counts[i] = np.sum(in_bin)
        
        if counts[i] > 0:
            accuracies[i] = np.mean(y_true_np[in_bin])
    
    return bin_centers, accuracies, counts


def precision_recall_at_k(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    k: int
) -> Tuple[float, float]:
    """
    Compute precision and recall at top-k predictions.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probability scores
        k: Number of top predictions to consider
        
    Returns:
        Tuple of (precision@k, recall@k)
    """
    # Get indices of top-k scores
    _, top_k_indices = torch.topk(y_scores, k)
    
    # Create predictions (1 for top-k, 0 otherwise)
    y_pred = torch.zeros_like(y_true)
    y_pred[top_k_indices] = 1
    
    # Compute metrics
    true_positives = torch.sum(y_true * y_pred).float()
    predicted_positives = torch.sum(y_pred).float()
    actual_positives = torch.sum(y_true).float()
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    
    return precision.item(), recall.item()


class MetricsTracker:
    """Track multiple metrics during training/evaluation."""
    
    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.metrics: dict[str, list[float]] = {}
    
    def update(self, **kwargs: float) -> None:
        """
        Update metrics with new values.
        
        Args:
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Average value
        """
        if metric_name not in self.metrics:
            raise KeyError(f"Metric {metric_name} not found")
        return np.mean(self.metrics[metric_name])
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
    
    def summary(self) -> dict[str, float]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary of metric averages
        """
        return {name: np.mean(values) for name, values in self.metrics.items()}