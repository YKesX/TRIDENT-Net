"""
Model calibration utilities for the TRIDENT system.

Implements temperature scaling and reliability diagram generation.
"""

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .metrics import expected_calibration_error, reliability_diagram_data


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    As described in "On Calibration of Modern Neural Networks" by Guo et al.
    """
    
    def __init__(self, temperature: float = 1.0) -> None:
        """
        Initialize temperature scaling.
        
        Args:
            temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits (B, C) or (B,)
            
        Returns:
            Temperature-scaled probabilities
        """
        if logits.dim() == 1:
            # Binary case
            scaled_logits = logits / self.temperature
            return torch.sigmoid(scaled_logits)
        else:
            # Multi-class case
            scaled_logits = logits / self.temperature
            return torch.softmax(scaled_logits, dim=1)


def calibrate_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01
) -> TemperatureScaling:
    """
    Calibrate model using temperature scaling.
    
    Args:
        model: Pre-trained model to calibrate
        val_loader: Validation data loader
        device: Device for computation
        max_iter: Maximum optimization iterations
        lr: Learning rate for temperature optimization
        
    Returns:
        Fitted TemperatureScaling module
    """
    model.eval()
    temperature_model = TemperatureScaling().to(device)
    optimizer = optim.LBFGS(
        temperature_model.parameters(), 
        lr=lr, 
        max_iter=max_iter
    )
    
    # Collect all logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"] if isinstance(batch, dict) else batch[0]
            labels = batch["labels"] if isinstance(batch, dict) else batch[1]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            if hasattr(logits, "p_outcome"):
                # Handle OutcomeEstimate objects
                logits = torch.logit(logits.p_outcome)
            
            all_logits.append(logits)
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Optimize temperature
    def eval_fn():
        optimizer.zero_grad()
        probs = temperature_model(all_logits)
        if probs.dim() == 2:
            loss = nn.CrossEntropyLoss()(all_logits / temperature_model.temperature, all_labels)
        else:
            loss = nn.BCELoss()(probs, all_labels.float())
        loss.backward()
        return loss
    
    optimizer.step(eval_fn)
    
    return temperature_model


class PlattScaling(nn.Module):
    """
    Platt scaling for binary classification calibration.
    
    Fits a sigmoid function to map scores to probabilities.
    """
    
    def __init__(self) -> None:
        """Initialize Platt scaling."""
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply Platt scaling to scores.
        
        Args:
            scores: Model scores (B,)
            
        Returns:
            Calibrated probabilities
        """
        return torch.sigmoid(self.a * scores + self.b)


def fit_platt_scaling(
    scores: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01
) -> PlattScaling:
    """
    Fit Platt scaling parameters.
    
    Args:
        scores: Model scores
        labels: Ground truth binary labels
        max_iter: Maximum optimization iterations
        lr: Learning rate
        
    Returns:
        Fitted PlattScaling module
    """
    platt_model = PlattScaling()
    optimizer = optim.Adam(platt_model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        probs = platt_model(scores)
        loss = criterion(probs, labels.float())
        loss.backward()
        optimizer.step()
    
    return platt_model


class IsotonicRegression:
    """
    Isotonic regression for non-parametric calibration.
    
    Fits a monotonic function to calibrate probabilities.
    """
    
    def __init__(self) -> None:
        """Initialize isotonic regression."""
        self.x_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicRegression":
        """
        Fit isotonic regression.
        
        Args:
            scores: Model scores
            labels: Ground truth binary labels
            
        Returns:
            Self for method chaining
        """
        from sklearn.isotonic import IsotonicRegression as SklearnIsotonic
        
        # Sort scores and labels
        sorted_indices = np.argsort(scores)
        self.x_ = scores[sorted_indices]
        
        # Fit isotonic regression
        iso_reg = SklearnIsotonic(out_of_bounds="clip")
        self.y_ = iso_reg.fit_transform(scores[sorted_indices], labels[sorted_indices])
        
        return self
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression calibration.
        
        Args:
            scores: Model scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if self.x_ is None or self.y_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return np.interp(scores, self.x_, self.y_)


def plot_reliability_diagram(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
) -> None:
    """
    Plot reliability diagram for calibration assessment.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        n_bins: Number of confidence bins
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    bin_centers, accuracies, counts = reliability_diagram_data(
        y_true, y_probs, n_bins
    )
    
    # Plot reliability diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability plot
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, accuracies, width=1.0/n_bins, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"{title} - Reliability")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(y_probs.detach().cpu().numpy(), bins=n_bins, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{title} - Distribution")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_calibration(
    y_true: torch.Tensor,
    y_probs: torch.Tensor
) -> dict[str, float]:
    """
    Evaluate calibration metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        
    Returns:
        Dictionary of calibration metrics
    """
    from .metrics import brier_score
    
    ece = expected_calibration_error(y_true, y_probs)
    brier = brier_score(y_true, y_probs)
    
    # Compute Maximum Calibration Error (MCE)
    bin_centers, accuracies, counts = reliability_diagram_data(y_true, y_probs)
    
    valid_bins = counts > 0
    if valid_bins.any():
        mce = np.max(np.abs(accuracies[valid_bins] - bin_centers[valid_bins]))
    else:
        mce = 0.0
    
    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier
    }


class CalibrationWrapper(nn.Module):
    """
    Wrapper that applies calibration to model outputs.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        calibrator: nn.Module
    ) -> None:
        """
        Initialize calibration wrapper.
        
        Args:
            model: Base model
            calibrator: Calibration module (e.g., TemperatureScaling)
        """
        super().__init__()
        self.model = model
        self.calibrator = calibrator
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with calibration.
        
        Returns:
            Calibrated output
        """
        output = self.model(*args, **kwargs)
        
        if hasattr(output, "p_outcome"):
            # Handle OutcomeEstimate objects
            logits = torch.logit(output.p_outcome)
            calibrated_probs = self.calibrator(logits)
            output.p_outcome = calibrated_probs
            return output
        else:
            # Direct probability output
            return self.calibrator(output)