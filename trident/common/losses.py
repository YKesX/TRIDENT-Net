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
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name](**kwargs)