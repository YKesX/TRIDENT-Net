"""
Loss functions for TRIDENT-Net.

Author: Yağızhan Keskin
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, H, W) or (B, H, W)
            target: Targets of shape (B, H, W)
        """
        if pred.dim() == 4 and pred.shape[1] > 1:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        if pred.dim() == 4:
            pred = pred[:, 1]  # Take positive class for binary
            
        pred = pred.flatten()
        target = target.flatten().float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C) or (B,)
            target: Targets of shape (B,)
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss."""
    
    def __init__(
        self, 
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


class IoULoss(nn.Module):
    """Intersection over Union Loss."""
    
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 4 and pred.shape[1] > 1:
            pred = torch.softmax(pred, dim=1)[:, 1]
        else:
            pred = torch.sigmoid(pred)
            
        pred = pred.flatten()
        target = target.flatten().float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class CEIoULoss(nn.Module):
    """Combined Cross Entropy + IoU Loss."""
    
    def __init__(self, ce_weight: float = 0.5, iou_weight: float = 0.5) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.iou_weight = iou_weight
        self.iou_loss = IoULoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target)
        iou = self.iou_loss(pred, target)
        return self.ce_weight * ce + self.iou_weight * iou


class HuberCELoss(nn.Module):
    """Combined Huber (for regression) + Cross Entropy (for classification)."""
    
    def __init__(
        self, 
        huber_weight: float = 0.5,
        ce_weight: float = 0.5,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.huber_weight = huber_weight
        self.ce_weight = ce_weight
        self.huber_loss = nn.HuberLoss(delta=delta)
    
    def forward(
        self, 
        reg_pred: torch.Tensor, 
        reg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
    ) -> torch.Tensor:
        huber = self.huber_loss(reg_pred, reg_target)
        ce = F.cross_entropy(cls_pred, cls_target)
        return self.huber_weight * huber + self.ce_weight * ce


class GIoULoss(nn.Module):
    """Generalized IoU Loss for object detection."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: Predicted boxes (x1, y1, x2, y2) shape (N, 4)
            target_boxes: Target boxes (x1, y1, x2, y2) shape (N, 4)
        """
        # Calculate intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Calculate enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Calculate GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
        
        return 1 - giou.mean()


class HierarchyRegularizer(nn.Module):
    """
    Hierarchy regularizer to enforce p_kill <= p_hit constraint.
    
    Computes λ * mean(relu(p_kill - p_hit)) to penalize violations
    of the logical hierarchy constraint.
    """
    
    def __init__(self, weight: float = 0.2):
        super().__init__()
        self.weight = weight
        
    def forward(self, p_hit: torch.Tensor, p_kill: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchy regularization loss.
        
        Args:
            p_hit: Hit probabilities (B, 1)
            p_kill: Kill probabilities (B, 1)
            
        Returns:
            Hierarchy regularization loss
        """
        # Penalize cases where p_kill > p_hit (logical violation)
        violations = torch.relu(p_kill - p_hit)
        return self.weight * violations.mean()


class BrierScore(nn.Module):
    """
    Brier Score for probability calibration assessment.
    
    Computes mean((p - y)^2) where p is predicted probability and y is binary target.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, p_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Brier score.
        
        Args:
            p_pred: Predicted probabilities (B, 1) in [0, 1]
            y_true: True binary labels (B, 1) in {0, 1}
            
        Returns:
            Brier score loss
        """
        return torch.mean((p_pred - y_true) ** 2)


class FusionMultitaskLoss(nn.Module):
    """
    Multitask loss for fusion module with hierarchy regularization.
    
    Combines:
    - BCE(hit) + BCE(kill) + Brier(hit) + Brier(kill) + λ*hierarchy_regularizer
    """
    
    def __init__(
        self, 
        bce_hit_weight: float = 1.0,
        bce_kill_weight: float = 1.0,
        brier_weight: float = 0.25,
        hierarchy_weight: float = 0.2
    ):
        super().__init__()
        
        self.bce_hit_weight = bce_hit_weight
        self.bce_kill_weight = bce_kill_weight
        self.brier_weight = brier_weight
        
        # Loss components
        self.bce_loss = nn.BCELoss()
        self.brier_score = BrierScore()
        self.hierarchy_regularizer = HierarchyRegularizer(hierarchy_weight)
        
    def forward(
        self, 
        p_hit: torch.Tensor, 
        p_kill: torch.Tensor,
        y_hit: torch.Tensor,
        y_kill: torch.Tensor
    ) -> dict:
        """
        Compute multitask fusion loss.
        
        Args:
            p_hit: Predicted hit probabilities (B, 1)
            p_kill: Predicted kill probabilities (B, 1)
            y_hit: True hit labels (B, 1)
            y_kill: True kill labels (B, 1)
            
        Returns:
            Dictionary with total loss and component losses
        """
        # Ensure probabilities are in [0, 1] and targets are float
        p_hit = torch.clamp(p_hit, 0.0, 1.0)
        p_kill = torch.clamp(p_kill, 0.0, 1.0)
        y_hit = y_hit.float()
        y_kill = y_kill.float()
        
        # Component losses
        bce_hit = self.bce_loss(p_hit, y_hit)
        bce_kill = self.bce_loss(p_kill, y_kill)
        brier_hit = self.brier_score(p_hit, y_hit)
        brier_kill = self.brier_score(p_kill, y_kill)
        hierarchy_reg = self.hierarchy_regularizer(p_hit, p_kill)
        
        # Total loss
        total_loss = (
            self.bce_hit_weight * bce_hit + 
            self.bce_kill_weight * bce_kill +
            self.brier_weight * brier_hit +
            self.brier_weight * brier_kill +
            hierarchy_reg
        )
        
        return {
            'total_loss': total_loss,
            'bce_hit': bce_hit,
            'bce_kill': bce_kill, 
            'brier_hit': brier_hit,
            'brier_kill': brier_kill,
            'hierarchy_reg': hierarchy_reg
        }


def get_loss_fn(loss_name: str, **kwargs) -> nn.Module:
    """Factory function for loss functions."""
    loss_map = {
        "dice": DiceLoss,
        "focal": FocalLoss,
        "dice_focal": DiceFocalLoss,
        "bce": nn.BCEWithLogitsLoss,
        "ce": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "ce_iou": CEIoULoss,
        "huber_ce": HuberCELoss,
        "giou_obj": GIoULoss,
        "hierarchy_regularizer": HierarchyRegularizer,
        "brier_score": BrierScore,
        "fusion_multitask": FusionMultitaskLoss,
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name](**kwargs)