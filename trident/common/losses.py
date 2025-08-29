"""
Loss functions for the TRIDENT system.

Implements various loss functions for multimodal learning and segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    
    Computes 1 - Dice coefficient as loss.
    """
    
    def __init__(self, smooth: float = 1e-5) -> None:
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Predicted probabilities (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
            
        Returns:
            Dice loss value
        """
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1.0 - dice_coeff


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    
    As described in "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = "mean"
    ) -> None:
        """
        Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            predictions: Predicted logits (B, C) or (B,)
            targets: Ground truth labels (B,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CrossEntropyIoULoss(nn.Module):
    """
    Combined Cross-Entropy and IoU loss for segmentation.
    """
    
    def __init__(
        self, 
        ce_weight: float = 1.0, 
        iou_weight: float = 1.0,
        smooth: float = 1e-5
    ) -> None:
        """
        Initialize combined loss.
        
        Args:
            ce_weight: Weight for cross-entropy term
            iou_weight: Weight for IoU term
            smooth: Smoothing factor for IoU
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined CE + IoU loss.
        
        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Combined loss value
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # IoU loss
        predictions_soft = F.softmax(predictions, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (predictions_soft * targets_one_hot).sum(dim=(2, 3))
        union = (predictions_soft + targets_one_hot).sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou.mean()
        
        return self.ce_weight * ce_loss + self.iou_weight * iou_loss


class BalancedBCELoss(nn.Module):
    """
    Balanced Binary Cross-Entropy loss.
    
    Automatically balances positive and negative samples.
    """
    
    def __init__(self, pos_weight: float = None) -> None:
        """
        Initialize balanced BCE loss.
        
        Args:
            pos_weight: Weight for positive samples (auto-computed if None)
        """
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute balanced BCE loss.
        
        Args:
            predictions: Predicted logits (B,) or (B, 1)
            targets: Ground truth labels (B,) or (B, 1)
            
        Returns:
            Balanced BCE loss value
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        if self.pos_weight is None:
            # Auto-compute positive weight
            num_pos = targets.sum()
            num_neg = (targets == 0).sum()
            if num_pos > 0:
                pos_weight = num_neg / num_pos
            else:
                pos_weight = 1.0
        else:
            pos_weight = self.pos_weight
        
        return F.binary_cross_entropy_with_logits(
            predictions, targets, pos_weight=torch.tensor(pos_weight)
        )


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similar/dissimilar representations.
    """
    
    def __init__(self, margin: float = 1.0) -> None:
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features1: First set of features (B, D)
            features2: Second set of features (B, D) 
            labels: Similarity labels (B,) - 1 for similar, 0 for dissimilar
            
        Returns:
            Contrastive loss value
        """
        # Compute euclidean distance
        distances = F.pairwise_distance(features1, features2)
        
        # Compute loss
        loss_similar = labels * torch.pow(distances, 2)
        loss_dissimilar = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        
        return torch.mean(loss_similar + loss_dissimilar)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable task weighting.
    
    Uses uncertainty-based weighting as in "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics" by Kendall et al.
    """
    
    def __init__(self, num_tasks: int) -> None:
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks
        """
        super().__init__()
        self.num_tasks = num_tasks
        # Learnable log-variance parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            losses: List of individual task losses
            
        Returns:
            Weighted multi-task loss
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")
        
        total_loss = 0.0
        for i, loss in enumerate(losses):
            # Precision weighting: 1 / (2 * sigma^2)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


def uncertainty_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor
) -> torch.Tensor:
    """
    Loss function that incorporates prediction uncertainty.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        uncertainty: Predicted uncertainty (log-variance)
        
    Returns:
        Uncertainty-weighted loss
    """
    # Base loss (e.g., MSE)
    base_loss = F.mse_loss(predictions, targets, reduction="none")
    
    # Uncertainty weighting: precision * loss + log-variance regularization
    precision = torch.exp(-uncertainty)
    weighted_loss = precision * base_loss + uncertainty
    
    return weighted_loss.mean()