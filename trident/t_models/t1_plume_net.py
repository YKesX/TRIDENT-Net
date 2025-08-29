"""
PlumeNet: Lightweight detector and tracker for thermal sources.

Combines anchor-free detection with simple tracking for thermal plume
detection and tracking in IR sequences.
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken, FeatureVec


class PlumeNet(BranchModule):
    """
    Plume detection and tracking network.
    
    Uses anchor-free detection head with lightweight tracking
    for thermal plume detection and velocity estimation.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        out_dim: int = 256,
        num_classes: int = 3,  # background, hotspot, plume
        max_detections: int = 100,
        tracking_memory: int = 10,
        pretrained: bool = True
    ) -> None:
        """
        Initialize PlumeNet.
        
        Args:
            backbone_name: Timm model name for backbone
            out_dim: Output feature dimension
            num_classes: Number of detection classes
            max_detections: Maximum detections per frame
            tracking_memory: Number of frames for tracking memory
            pretrained: Use pretrained backbone weights
        """
        super().__init__()
        
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.max_detections = max_detections
        self.tracking_memory = tracking_memory
        
        # Lightweight backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4)  # Multi-scale features
        )
        
        # Get backbone feature dimensions
        dummy_input = torch.randn(1, 1, 224, 224)  # Single channel IR
        with torch.no_grad():
            backbone_features = self.backbone(dummy_input)
        self.backbone_dims = [f.shape[1] for f in backbone_features]
        
        # Feature Pyramid Network for multi-scale detection
        self.fpn = FeaturePyramidNetwork(self.backbone_dims)
        
        # Anchor-free detection head
        self.detection_head = AnchorFreeDetectionHead(
            feature_dim=256,  # FPN output dimension
            num_classes=num_classes,
            num_levels=len(self.backbone_dims)
        )
        
        # Feature extraction for tracking
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone_dims[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
        
        # Simple tracker
        self.tracker = SimpleTracker(
            feature_dim=out_dim,
            memory_size=tracking_memory
        )
        
        # Plume analysis modules
        self.plume_analyzer = PlumeAnalyzer(out_dim)
        
        # Track storage for temporal consistency
        self.track_history: List[Dict[str, Any]] = []
    
    def forward(self, ir_roi_t: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process IR temporal sequence for plume detection and tracking.
        
        Args:
            ir_roi_t: IR temporal sequence (B, T, 1, H, W)
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size, seq_len, channels, height, width = ir_roi_t.shape
        
        # Process each frame for detection
        all_detections = []
        all_features = []
        
        for t in range(seq_len):
            frame = ir_roi_t[:, t]  # (B, 1, H, W)
            
            # Backbone feature extraction
            backbone_features = self.backbone(frame)
            
            # FPN for multi-scale features
            fpn_features = self.fpn(backbone_features)
            
            # Detection
            detections = self.detection_head(fpn_features)
            all_detections.append(detections)
            
            # Global feature extraction
            global_features = self.feature_extractor(backbone_features[-1])
            all_features.append(global_features)
        
        # Stack temporal features
        temporal_features = torch.stack(all_features, dim=1)  # (B, T, out_dim)
        
        # Aggregate temporal features
        aggregated_features = torch.mean(temporal_features, dim=1)  # (B, out_dim)
        
        # Create feature vector
        feature_vec = FeatureVec(z=aggregated_features)
        
        # Track detections across time
        events = []
        for b in range(batch_size):
            batch_detections = [det[b] for det in all_detections]
            batch_features = temporal_features[b]  # (T, out_dim)
            
            # Update tracker and get tracks
            tracks = self.tracker.update(batch_detections, batch_features)
            
            # Analyze plumes and generate events
            plume_events = self.plume_analyzer.analyze_tracks(
                tracks, seq_len, b
            )
            events.extend(plume_events)
        
        return feature_vec, events
    
    def get_detections(self, ir_roi_t: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Get raw detections for visualization.
        
        Args:
            ir_roi_t: IR temporal sequence (B, T, 1, H, W)
            
        Returns:
            List of detections per frame
        """
        batch_size, seq_len, channels, height, width = ir_roi_t.shape
        all_detections = []
        
        for t in range(seq_len):
            frame = ir_roi_t[:, t]
            backbone_features = self.backbone(frame)
            fpn_features = self.fpn(backbone_features)
            detections = self.detection_head(fpn_features)
            all_detections.append(detections)
        
        return all_detections


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale detection."""
    
    def __init__(self, backbone_dims: List[int], feature_dim: int = 256) -> None:
        """
        Initialize FPN.
        
        Args:
            backbone_dims: Backbone feature dimensions
            feature_dim: FPN feature dimension
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, feature_dim, 1) for dim in backbone_dims
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1) 
            for _ in backbone_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through FPN.
        
        Args:
            features: List of backbone features (low to high resolution)
            
        Returns:
            List of FPN features
        """
        # Lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in 
                   zip(self.lateral_convs, features)]
        
        # Top-down pathway
        fpn_features = []
        
        # Start from highest level (lowest resolution)
        prev_feat = laterals[-1]
        fpn_features.append(self.fpn_convs[-1](prev_feat))
        
        # Process remaining levels
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample previous feature
            upsampled = F.interpolate(
                prev_feat, size=laterals[i].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            
            # Add lateral connection
            merged = laterals[i] + upsampled
            
            # Apply FPN conv
            fpn_feat = self.fpn_convs[i](merged)
            fpn_features.insert(0, fpn_feat)
            
            prev_feat = merged
        
        return fpn_features


class AnchorFreeDetectionHead(nn.Module):
    """Anchor-free detection head for thermal sources."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 3,
        num_levels: int = 3
    ) -> None:
        """
        Initialize detection head.
        
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of classes
            num_levels: Number of FPN levels
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_levels = num_levels
        
        # Shared convolutions
        self.shared_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Conv2d(feature_dim, num_classes, 3, padding=1)
        
        # Regression head (center offset + size)
        self.reg_head = nn.Conv2d(feature_dim, 4, 3, padding=1)
        
        # Centerness head
        self.centerness_head = nn.Conv2d(feature_dim, 1, 3, padding=1)
    
    def forward(self, fpn_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through detection head.
        
        Args:
            fpn_features: List of FPN features
            
        Returns:
            Dictionary of detection outputs
        """
        cls_outputs = []
        reg_outputs = []
        centerness_outputs = []
        
        for fpn_feat in fpn_features:
            # Shared convolutions
            shared_feat = self.shared_conv(fpn_feat)
            
            # Classification
            cls_output = self.cls_head(shared_feat)
            cls_outputs.append(cls_output)
            
            # Regression
            reg_output = self.reg_head(shared_feat)
            reg_outputs.append(reg_output)
            
            # Centerness
            centerness_output = torch.sigmoid(self.centerness_head(shared_feat))
            centerness_outputs.append(centerness_output)
        
        return {
            'cls_outputs': cls_outputs,
            'reg_outputs': reg_outputs,
            'centerness_outputs': centerness_outputs
        }


class SimpleTracker(nn.Module):
    """Simple tracker for thermal sources using feature similarity."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        memory_size: int = 10,
        similarity_threshold: float = 0.7
    ) -> None:
        """
        Initialize simple tracker.
        
        Args:
            feature_dim: Feature dimension
            memory_size: Number of frames to remember
            similarity_threshold: Threshold for track association
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        
        self.tracks: List[Dict[str, Any]] = []
        self.next_track_id = 0
    
    def update(
        self,
        detections: List[Dict[str, torch.Tensor]],
        features: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections per frame
            features: Features per frame (T, feature_dim)
            
        Returns:
            Updated tracks
        """
        # Simple implementation: just store detections with features
        # In a full implementation, this would do proper tracking
        
        seq_len = len(detections)
        updated_tracks = []
        
        for t, detection in enumerate(detections):
            if detection and len(detection.get('cls_outputs', [])) > 0:
                # Create track for this detection
                track = {
                    'track_id': self.next_track_id,
                    'start_time': t,
                    'end_time': t,
                    'features': features[t:t+1],
                    'detections': [detection],
                    'positions': [],  # Would extract from detection
                    'confidence': 0.8,  # Simplified
                }
                updated_tracks.append(track)
                self.next_track_id += 1
        
        return updated_tracks


class PlumeAnalyzer(nn.Module):
    """Analyze plume tracks for event generation."""
    
    def __init__(self, feature_dim: int) -> None:
        """
        Initialize plume analyzer.
        
        Args:
            feature_dim: Feature dimension
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Velocity estimator
        self.velocity_estimator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # x, y velocity
            nn.Tanh()
        )
        
        # Intensity analyzer
        self.intensity_analyzer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def analyze_tracks(
        self,
        tracks: List[Dict[str, Any]],
        seq_len: int,
        batch_idx: int
    ) -> List[EventToken]:
        """
        Analyze tracks and generate events.
        
        Args:
            tracks: List of tracks
            seq_len: Sequence length
            batch_idx: Batch index
            
        Returns:
            List of event tokens
        """
        events = []
        
        for track in tracks:
            if len(track['features']) == 0:
                continue
                
            # Analyze track features
            track_features = track['features'].mean(dim=0)  # Average features
            
            # Estimate velocity
            velocity = self.velocity_estimator(track_features)
            velocity_magnitude = torch.norm(velocity).item()
            
            # Analyze intensity
            intensity = self.intensity_analyzer(track_features).item()
            
            # Generate hotspot event
            if intensity > 0.6:
                events.append(EventToken(
                    type="hotspot_count",
                    value=1.0,  # Count of hotspots
                    t_start=track['start_time'],
                    t_end=track['end_time'] + 1,
                    quality=intensity,
                    meta={
                        "source": "plume_net",
                        "batch_idx": batch_idx,
                        "track_id": track['track_id'],
                        "intensity": intensity
                    }
                ))
            
            # Generate plume velocity event
            if velocity_magnitude > 0.3:
                events.append(EventToken(
                    type="plume_velocity",
                    value=velocity_magnitude,
                    t_start=track['start_time'],
                    t_end=track['end_time'] + 1,
                    quality=min(1.0, velocity_magnitude),
                    meta={
                        "source": "plume_net",
                        "batch_idx": batch_idx,
                        "track_id": track['track_id'],
                        "velocity_x": velocity[0].item(),
                        "velocity_y": velocity[1].item(),
                        "velocity_magnitude": velocity_magnitude
                    }
                ))
        
        return events


# Default configuration
PLUME_NET_CONFIG = {
    "backbone_name": "efficientnet_b0",
    "out_dim": 256,
    "num_classes": 3,
    "max_detections": 100,
    "tracking_memory": 10,
    "pretrained": True
}