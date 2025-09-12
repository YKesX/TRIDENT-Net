"""
TRIDENT-T1: PlumeDetLite - Lightweight plume detection and tracking over 3 IR frames

DEPRECATED: This module is legacy v1 code. Use trident_t.ir_dettrack_v2.PlumeDetXL instead.

Author: Yağızhan Keskin
"""

import warnings
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken

# Issue deprecation warning when module is imported
warnings.warn(
    "trident_t.plumedet_lite is deprecated. Use trident_t.ir_dettrack_v2.PlumeDetXL instead.",
    DeprecationWarning,
    stacklevel=2
)


class PlumeDetLite(BranchModule):
    """
    Tiny anchor-free detector per IR frame + simple association over 3 frames.
    
    Detects plumes/hotspots in each IR frame and associates them across time
    to form tracks for thermal analysis.
    
    Input: ir_seq (B, 3, 1, 480, 640) - 3 frames of IR
    Outputs:
        - tracks (list[track]) - variable length per sample
        - zt (B, 256) - pooled embedding
        - events (list) - detected thermal events
    """
    
    def __init__(self, det_backbone: str = "lite", max_tracks: int = 10, out_dim: int = 256):
        super().__init__(out_dim=out_dim)
        
        self.max_tracks = max_tracks
        
        # Lightweight detection backbone
        self.backbone = LiteDetectionBackbone()
        
        # Detection heads (anchor-free)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # x, y offsets
        )
        
        self.size_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),  # width, height
            nn.ReLU(inplace=True)
        )
        
        self.intensity_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_proj = nn.Linear(128, out_dim)
        
        # Simple tracker
        self.tracker = SimpleTracker(max_tracks=max_tracks)
        
    def forward(self, ir_seq: torch.Tensor) -> Tuple[List[Dict], torch.Tensor, List[EventToken]]:
        """
        Forward pass through PlumeDetLite.
        
        Args:
            ir_seq: IR sequence (B, 3, 1, 480, 640)
            
        Returns:
            tuple: (tracks, zt, events)
                - tracks: List of track dictionaries
                - zt: (B, 256) feature embedding
                - events: List of detected thermal events
        """
        B, T, C, H, W = ir_seq.shape
        
        # Process each frame
        frame_detections = []
        frame_features = []
        
        for t in range(T):
            frame = ir_seq[:, t]  # (B, 1, H, W)
            
            # Extract features
            features = self.backbone(frame)  # (B, 128, H', W')
            frame_features.append(features)
            
            # Detection heads
            heatmap = self.heatmap_head(features)  # (B, 1, H', W')
            offsets = self.offset_head(features)   # (B, 2, H', W')
            sizes = self.size_head(features)       # (B, 2, H', W')
            intensities = self.intensity_head(features)  # (B, 1, H', W')
            
            # Extract detections from heatmap
            detections = self._extract_detections(heatmap, offsets, sizes, intensities, t)
            frame_detections.append(detections)
        
        # Average features across time for embedding
        avg_features = torch.stack(frame_features, dim=1).mean(dim=1)  # (B, 128, H', W')
        pooled = self.global_pool(avg_features).view(B, -1)  # (B, 128)
        zt = self.embedding_proj(pooled)  # (B, out_dim)
        
        # Track detections across frames
        tracks = self.tracker.track_sequence(frame_detections)
        
        # Generate events from tracks
        events = self._extract_thermal_events(tracks)
        
        return tracks, zt, events
    
    def _extract_detections(self, heatmap: torch.Tensor, offsets: torch.Tensor, 
                          sizes: torch.Tensor, intensities: torch.Tensor, timestamp: int) -> List[Dict]:
        """Extract detections from network outputs."""
        B, _, H, W = heatmap.shape
        detections = []
        
        threshold = 0.5
        
        for b in range(B):
            hm = heatmap[b, 0]  # (H, W)
            
            # Find peaks
            peaks = self._find_peaks(hm, threshold)
            
            for peak_y, peak_x in peaks:
                # Get prediction values at peak
                score = hm[peak_y, peak_x].item()
                offset_x = offsets[b, 0, peak_y, peak_x].item()
                offset_y = offsets[b, 1, peak_y, peak_x].item()
                width = sizes[b, 0, peak_y, peak_x].item()
                height = sizes[b, 1, peak_y, peak_x].item()
                intensity = intensities[b, 0, peak_y, peak_x].item()
                
                # Convert to original image coordinates
                scale_x = 640 / W
                scale_y = 480 / H
                
                center_x = (peak_x + offset_x) * scale_x
                center_y = (peak_y + offset_y) * scale_y
                bbox_w = width * scale_x
                bbox_h = height * scale_y
                
                detection = {
                    'batch_idx': b,
                    'timestamp': timestamp,
                    'score': score,
                    'bbox': [center_x - bbox_w/2, center_y - bbox_h/2, bbox_w, bbox_h],  # x, y, w, h
                    'center': [center_x, center_y],
                    'intensity': intensity,
                    'area': bbox_w * bbox_h
                }
                detections.append(detection)
        
        return detections
    
    def _find_peaks(self, heatmap: torch.Tensor, threshold: float) -> List[Tuple[int, int]]:
        """Find local peaks in heatmap."""
        H, W = heatmap.shape
        peaks = []
        
        # Simple peak finding with 3x3 neighborhood
        for y in range(1, H-1):
            for x in range(1, W-1):
                if heatmap[y, x] > threshold:
                    # Check if it's a local maximum
                    is_peak = True
                    center_val = heatmap[y, x]
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if heatmap[y + dy, x + dx] >= center_val:
                                is_peak = False
                                break
                        if not is_peak:
                            break
                    
                    if is_peak:
                        peaks.append((y, x))
        
        return peaks
    
    def _extract_thermal_events(self, tracks: List[Dict]) -> List[EventToken]:
        """Extract thermal events from tracks."""
        events = []
        
        for track in tracks:
            if len(track['detections']) >= 2:  # At least 2 detections for a track
                # Compute track statistics
                intensities = [det['intensity'] for det in track['detections']]
                areas = [det['area'] for det in track['detections']]
                
                avg_intensity = sum(intensities) / len(intensities)
                max_intensity = max(intensities)
                total_area = sum(areas) / len(areas)
                
                # Determine event type based on thermal signature
                if max_intensity > 0.8:
                    event_type = "hot_plume"
                elif avg_intensity > 0.5:
                    event_type = "thermal_signature"
                else:
                    event_type = "weak_thermal"
                
                # Use last detection for location
                last_det = track['detections'][-1]
                
                event = EventToken(
                    event_type=event_type,
                    confidence=max_intensity,
                    location=tuple(last_det['center']),
                    timestamp=last_det['timestamp'],
                    source="plumedetlite",
                    metadata={
                        'track_id': track['id'],
                        'avg_intensity': avg_intensity,
                        'max_intensity': max_intensity,
                        'total_area': total_area,
                        'track_length': len(track['detections']),
                        'batch_idx': last_det['batch_idx']
                    }
                )
                events.append(event)
        
        return events


class LiteDetectionBackbone(nn.Module):
    """Lightweight detection backbone for IR frames."""
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SimpleTracker:
    """Simple association tracker for thermal detections."""
    
    def __init__(self, max_tracks: int = 10, distance_threshold: float = 50.0):
        self.max_tracks = max_tracks
        self.distance_threshold = distance_threshold
        self.next_id = 0
    
    def track_sequence(self, frame_detections: List[List[Dict]]) -> List[Dict]:
        """Track detections across a sequence of frames."""
        tracks = []
        
        # Initialize tracks from first frame
        if frame_detections and frame_detections[0]:
            for det in frame_detections[0]:
                track = {
                    'id': self.next_id,
                    'detections': [det],
                    'active': True
                }
                tracks.append(track)
                self.next_id += 1
        
        # Associate subsequent frames
        for frame_idx in range(1, len(frame_detections)):
            frame_dets = frame_detections[frame_idx]
            
            # Match detections to existing tracks
            matched_tracks = set()
            matched_dets = set()
            
            for track_idx, track in enumerate(tracks):
                if not track['active']:
                    continue
                    
                last_det = track['detections'][-1]
                last_pos = last_det['center']
                
                best_det_idx = None
                best_distance = float('inf')
                
                for det_idx, det in enumerate(frame_dets):
                    if det_idx in matched_dets:
                        continue
                    
                    # Only match detections from same batch
                    if det['batch_idx'] != last_det['batch_idx']:
                        continue
                    
                    # Compute distance
                    curr_pos = det['center']
                    distance = ((last_pos[0] - curr_pos[0])**2 + (last_pos[1] - curr_pos[1])**2)**0.5
                    
                    if distance < self.distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_det_idx = det_idx
                
                if best_det_idx is not None:
                    track['detections'].append(frame_dets[best_det_idx])
                    matched_tracks.add(track_idx)
                    matched_dets.add(best_det_idx)
                else:
                    track['active'] = False
            
            # Create new tracks for unmatched detections
            for det_idx, det in enumerate(frame_dets):
                if det_idx not in matched_dets and len(tracks) < self.max_tracks:
                    track = {
                        'id': self.next_id,
                        'detections': [det],
                        'active': True
                    }
                    tracks.append(track)
                    self.next_id += 1
        
        # Filter out tracks with too few detections
        valid_tracks = [track for track in tracks if len(track['detections']) >= 1]
        
        return valid_tracks