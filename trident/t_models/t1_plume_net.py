"""
TRIDENT-T1: PlumeNet (detection and tracking)

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken, FeatureVec


class AnchorFreeDetector(nn.Module):
    """Anchor-free object detector for thermal signatures."""
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Detection heads
        self.objectness_head = nn.Conv2d(128, 1, 3, 1, 1)  # Object/background
        self.center_head = nn.Conv2d(128, 2, 3, 1, 1)     # Center offset
        self.size_head = nn.Conv2d(128, 2, 3, 1, 1)       # Width, height
        self.class_head = nn.Conv2d(128, num_classes, 3, 1, 1)  # Class scores
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for detection.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict with detection outputs
        """
        features = self.backbone(x)
        
        # Generate predictions
        objectness = torch.sigmoid(self.objectness_head(features))
        center_offset = self.center_head(features)
        size_pred = F.relu(self.size_head(features))
        class_scores = torch.sigmoid(self.class_head(features))
        
        return {
            'objectness': objectness,
            'center_offset': center_offset,
            'size': size_pred,
            'class_scores': class_scores,
            'features': features,
        }


class SimpleTracker(nn.Module):
    """Simple Kalman-filter-like tracker for thermal signatures."""
    
    def __init__(self, max_tracks: int = 10):
        super().__init__()
        self.max_tracks = max_tracks
        self.tracks = []
        self.track_id_counter = 0
    
    def update(self, detections: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of updated tracks
        """
        # Simple assignment based on distance (in practice, use Hungarian algorithm)
        updated_tracks = []
        
        for det in detections:
            best_match = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track in self.tracks:
                if track['active']:
                    distance = ((det['x'] - track['x'])**2 + (det['y'] - track['y'])**2)**0.5
                    if distance < best_distance and distance < 50:  # Threshold
                        best_distance = distance
                        best_match = track
            
            if best_match:
                # Update existing track
                alpha = 0.3  # Smoothing factor
                best_match['x'] = alpha * det['x'] + (1 - alpha) * best_match['x']
                best_match['y'] = alpha * det['y'] + (1 - alpha) * best_match['y']
                best_match['confidence'] = max(best_match['confidence'], det['confidence'])
                best_match['age'] += 1
                updated_tracks.append(best_match)
            else:
                # Create new track
                if len(self.tracks) < self.max_tracks:
                    new_track = {
                        'id': self.track_id_counter,
                        'x': det['x'],
                        'y': det['y'],
                        'confidence': det['confidence'],
                        'age': 0,
                        'active': True,
                    }
                    self.track_id_counter += 1
                    self.tracks.append(new_track)
                    updated_tracks.append(new_track)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] < 10]
        
        return updated_tracks
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.track_id_counter = 0


class PlumeNet(BranchModule):
    """
    PlumeNet: Detection and tracking of thermal signatures.
    
    Combines lightweight anchor-free detection with simple tracking
    to monitor thermal plumes and hotspots over time.
    """
    
    def __init__(
        self,
        det_backbone: str = "lite",
        flow_head: str = "raft_lite",
        out_dim: int = 256,
        max_tracks: int = 10,
        detection_threshold: float = 0.5,
    ):
        super().__init__(out_dim)
        
        self.det_backbone = det_backbone
        self.flow_head = flow_head
        self.max_tracks = max_tracks
        self.detection_threshold = detection_threshold
        
        # Detector
        self.detector = AnchorFreeDetector(in_channels=1, num_classes=1)
        
        # Tracker
        self.tracker = SimpleTracker(max_tracks=max_tracks)
        
        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1, 16, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 8, 8)),
        )
        
        # Global feature head
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim),
        )
        
        # Event analysis heads
        self.plume_velocity_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # vx, vy
        )
        
        self.hotspot_counter = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.thermal_intensity_head = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, ir_roi_t: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for thermal detection and tracking.
        
        Args:
            ir_roi_t: Thermal IR sequence of shape (B, T, 1, H, W)
                     
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        B, T, C, H, W = ir_roi_t.shape
        
        # Process temporal features
        temporal_features = self.temporal_conv(ir_roi_t)  # (B, 32, 1, 8, 8)
        global_features = self.feature_head(temporal_features)  # (B, out_dim)
        
        # Process each frame for detection
        all_detections = []
        detection_outputs = []
        
        for t in range(T):
            frame = ir_roi_t[:, t]  # (B, 1, H, W)
            det_output = self.detector(frame)
            detection_outputs.append(det_output)
            
            # Extract detections for tracking
            for b in range(B):
                objectness = det_output['objectness'][b, 0]  # (H', W')
                center_offset = det_output['center_offset'][b]  # (2, H', W')
                size_pred = det_output['size'][b]  # (2, H', W')
                
                # Find detection candidates
                obj_mask = objectness > self.detection_threshold
                if obj_mask.sum() > 0:
                    # Get detection coordinates
                    y_coords, x_coords = torch.where(obj_mask)
                    
                    detections = []
                    for i in range(len(y_coords)):
                        y, x = y_coords[i].item(), x_coords[i].item()
                        
                        # Scale coordinates back to original resolution
                        scale_y, scale_x = H / objectness.shape[0], W / objectness.shape[1]
                        
                        detection = {
                            'x': x * scale_x + center_offset[0, y, x].item(),
                            'y': y * scale_y + center_offset[1, y, x].item(),
                            'w': size_pred[0, y, x].item(),
                            'h': size_pred[1, y, x].item(),
                            'confidence': objectness[y, x].item(),
                            'frame': t,
                            'batch': b,
                        }
                        detections.append(detection)
                    
                    all_detections.extend(detections)
        
        # Update tracks (simplified for batch processing)
        tracks = self.tracker.update([d for d in all_detections if d['batch'] == 0])
        
        # Event analysis
        plume_velocity = self.plume_velocity_head(global_features)
        hotspot_count = self.hotspot_counter(global_features)
        thermal_intensity = self.thermal_intensity_head(global_features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=global_features)
        
        # Create EventTokens
        events = []
        for b in range(B):
            # Track count event
            active_tracks = len([t for t in tracks if t['active']])
            if active_tracks > 0:
                events.append(EventToken(
                    type="hotspot_count",
                    value=float(active_tracks),
                    t_start=0.0,
                    t_end=T / 30.0,  # Assume 30 FPS
                    quality=min(active_tracks / 5.0, 1.0),
                    meta={
                        "active_tracks": active_tracks,
                        "total_detections": len([d for d in all_detections if d['batch'] == b]),
                        "avg_confidence": sum(t['confidence'] for t in tracks) / max(len(tracks), 1),
                        "temporal_frames": T,
                    }
                ))
            
            # Plume velocity event
            vx, vy = plume_velocity[b, 0].item(), plume_velocity[b, 1].item()
            velocity_magnitude = (vx**2 + vy**2)**0.5
            
            if velocity_magnitude > 1.0:  # pixels/frame threshold
                events.append(EventToken(
                    type="plume_velocity",
                    value=velocity_magnitude,
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=min(velocity_magnitude / 10.0, 1.0),
                    meta={
                        "velocity_x": vx,
                        "velocity_y": vy,
                        "velocity_magnitude": velocity_magnitude,
                        "velocity_pixels_per_frame": velocity_magnitude,
                    }
                ))
            
            # Thermal intensity event
            intensity = thermal_intensity[b, 0].item()
            if intensity > 0.5:
                events.append(EventToken(
                    type="thermal_intensity",
                    value=intensity,
                    t_start=0.0,
                    t_end=T / 30.0,
                    quality=min(intensity, 1.0),
                    meta={
                        "intensity_level": intensity,
                        "intensity_category": "high" if intensity > 0.8 else "moderate",
                    }
                ))
        
        # Store intermediate outputs
        self._last_detections = all_detections
        self._last_tracks = tracks
        self._last_detection_outputs = detection_outputs
        self._last_temporal_features = temporal_features
        
        return feature_vec, events
    
    def get_detections(self) -> List[Dict[str, Any]]:
        """Get all detections from last forward pass."""
        if hasattr(self, '_last_detections'):
            return self._last_detections
        else:
            raise RuntimeError("No detections available. Run forward() first.")
    
    def get_tracks(self) -> List[Dict[str, Any]]:
        """Get active tracks from last forward pass."""
        if hasattr(self, '_last_tracks'):
            return self._last_tracks
        else:
            raise RuntimeError("No tracks available. Run forward() first.")
    
    def get_detection_outputs(self) -> List[Dict[str, torch.Tensor]]:
        """Get raw detection outputs for each frame."""
        if hasattr(self, '_last_detection_outputs'):
            return self._last_detection_outputs
        else:
            raise RuntimeError("No detection outputs available. Run forward() first.")
    
    def reset_tracker(self):
        """Reset tracker state."""
        self.tracker.reset()


def create_plume_net(config: dict) -> PlumeNet:
    """Factory function to create PlumeNet from config."""
    return PlumeNet(
        det_backbone=config.get("det_backbone", "lite"),
        flow_head=config.get("flow_head", "raft_lite"),
        out_dim=config.get("out_dim", 256),
        max_tracks=config.get("max_tracks", 10),
        detection_threshold=config.get("detection_threshold", 0.5),
    )