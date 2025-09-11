"""
Model calibration utilities for TRIDENT-Net.

Author: Yağızhan Keskin
"""

from typing import Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    Figure = plt.Figure
except ImportError:
    plt = None
    Figure = Any
    
try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    class IsotonicRegression:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return X


class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration."""
    
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """
        Fit temperature parameter using validation data.
        
        Args:
            logits: Model logits before softmax
            labels: Ground truth labels
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
        """
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_fn():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        
        # Ensure temperature is positive
        self.temperature.data = torch.clamp(self.temperature.data, min=1e-3)


class PlattScaling:
    """Platt scaling for binary classification calibration."""
    
    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit Platt scaling parameters.
        
        Args:
            scores: Model output scores
            labels: Binary ground truth labels
        """
        try:
            from sklearn.linear_model import LogisticRegression
            # Reshape for sklearn
            scores = scores.reshape(-1, 1)
            
            # Fit logistic regression
            lr = LogisticRegression()
            lr.fit(scores, labels)
            
            self.a = lr.coef_[0][0]
            self.b = lr.intercept_[0]
        except ImportError:
            # Simple fallback implementation
            self.a = 1.0
            self.b = 0.0
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling transformation."""
        return 1.0 / (1.0 + np.exp(self.a * scores + self.b))


class IsotonicCalibration:
    """Isotonic regression for calibration."""
    
    def __init__(self) -> None:
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic calibration."""
        self.calibrator.fit(scores, labels)
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        return self.calibrator.transform(scores)


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    return_data: bool = False,
) -> Tuple[Figure, dict]:
    """
    Create reliability diagram for calibration assessment.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        strategy: Binning strategy ('uniform' or 'quantile')
        return_data: Whether to return raw data
        
    Returns:
        tuple: (matplotlib figure, calibration data dict)
    """
    # Create bins
    if strategy == "uniform":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
    else:
        raise ValueError("Strategy must be 'uniform' or 'quantile'")
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate calibration metrics per bin
    bin_means = []
    bin_accs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            bin_means.append(avg_confidence_in_bin)
            bin_accs.append(accuracy_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_means.append(0)
            bin_accs.append(0)
            bin_counts.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.bar(
        bin_means, 
        bin_accs, 
        width=1.0/n_bins, 
        alpha=0.7, 
        edgecolor='black',
        label='Model'
    )
    
    # Add bin counts as text
    for i, (mean, acc, count) in enumerate(zip(bin_means, bin_accs, bin_counts)):
        if count > 0:
            ax.text(mean, acc + 0.05, str(count), ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Prepare return data
    calib_data = {
        'bin_means': bin_means,
        'bin_accs': bin_accs, 
        'bin_counts': bin_counts,
        'bin_boundaries': bin_boundaries,
    }
    
    if return_data:
        return fig, calib_data
    else:
        return fig, calib_data


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.
    
    Returns:
        tuple: (fraction_of_positives, mean_predicted_value)
    """
    if strategy == "uniform":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError("Strategy must be 'uniform' or 'quantile'")
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    fraction_of_positives = []
    mean_predicted_value = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            fraction_of_positives.append(y_true[in_bin].mean())
            mean_predicted_value.append(y_prob[in_bin].mean())
        else:
            fraction_of_positives.append(0)
            mean_predicted_value.append((bin_lower + bin_upper) / 2)
    
    return np.array(fraction_of_positives), np.array(mean_predicted_value)