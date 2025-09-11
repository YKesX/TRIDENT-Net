"""
Anchor-free IR detector with tracking for TRIDENT-Net T-branch.

Detects thermal signatures per frame and associates across time.

Author: Yağızhan Keskin
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import EventToken


class Track:
    """Simple track object for thermal detections."""
    
    def __init__(self, track_id: int):
        self.id = track_id
        self.t_idx: List[int] = []
        self.center_xy: List[Tuple[float, float]] = []
        self.wh: List[Tuple[float, float]] = []
        self.score: List[float] = []
        self.intensity: List[float] = []
        self.active = True
        self.last_update = -1
    
    def add_detection(
        self,
        t_idx: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        score: float,
        intensity: float
    ):
        """Add detection to track."""
        self.t_idx.append(t_idx)
        self.center_xy.append((center_x, center_y))
        self.wh.append((width, height))
        self.score.append(score)
        self.intensity.append(intensity)
        self.last_update = t_idx
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary representation."""
        return {
            'id': self.id,
            't_idx': self.t_idx,
            'center_xy': self.center_xy,
            'wh': self.wh,
            'score': self.score,
            'intensity': self.intensity
        }


class PlumeDetXL(nn.Module):
    """
    Anchor-free IR detector with nearest-neighbor tracking.
    
    Processes IR video clips and generates:
    - Per-frame detections with heatmaps
    - Tracks associating detections across time
    - Global embedding vector
    - Event tokens from detection patterns
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        backbone: Dict[str, Any] = None,
        heads: Dict[str, Any] = None,
        tracker: Dict[str, Any] = None,
        pool_to_embed: int = 256
    ):
        """
        Initialize PlumeDetXL.
        
        Args:
            in_channels: Input channels (1 for IR)
            backbone: Backbone configuration
            heads: Detection head configuration
            tracker: Tracker configuration  
            pool_to_embed: Output embedding dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.pool_to_embed = pool_to_embed
        
        # Backbone config defaults
        if backbone is None:
            backbone = {}
        self.backbone_cfg = {
            'type': 'lite_cnn',
            'width': 128,
            'depth': 3,
            'fpn': True,
            **backbone
        }
        
        # Detection heads config
        if heads is None:
            heads = {}
        self.heads_cfg = {
            'heatmap_thr': 0.45,
            'max_dets_per_frame': 24,
            **heads
        }
        
        # Tracker config
        if tracker is None:
            tracker = {}
        self.tracker_cfg = {
            'type': 'nearest',
            'max_tracks': 16,
            'max_frame_gap': 1,
            **tracker
        }
        
        # Build backbone
        self.backbone = self._build_backbone()
        self.backbone_channels = self._get_backbone_channels()
        
        # Build FPN if enabled
        if self.backbone_cfg['fpn']:
            self.fpn = self._build_fpn()
            self.feature_channels = self.backbone_cfg['width']
        else:
            self.fpn = None
            self.feature_channels = self.backbone_channels[-1]
        
        # Detection heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(self.feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(self.feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 2, 2, 1)
        )
        
        self.size_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(self.feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 2, 2, 1),
            nn.ReLU()  # Ensure positive sizes
        )
        
        self.intensity_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(self.feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels // 2, 1, 1),
            nn.ReLU()  # Positive intensity values
        )
        
        # Global pooling and embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_proj = nn.Linear(self.feature_channels, pool_to_embed)
        
        # Track ID counter
        self.next_track_id = 0
        
    def _build_backbone(self) -> nn.Module:
        """Build backbone network."""
        cfg = self.backbone_cfg
        
        if cfg['type'] == 'lite_cnn':
            return self._build_lite_cnn()
        else:
            raise ValueError(f"Unknown backbone type: {cfg['type']}")
    
    def _build_lite_cnn(self) -> nn.Module:
        """Build lightweight CNN backbone."""
        cfg = self.backbone_cfg
        width = cfg['width']
        depth = cfg['depth']
        
        layers = []
        in_ch = self.in_channels
        
        # Initial conv
        layers.append(nn.Conv2d(in_ch, width // 4, 7, stride=2, padding=3))
        layers.append(nn.BatchNorm2d(width // 4))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        
        in_ch = width // 4
        
        # Build depth blocks
        for i in range(depth):
            out_ch = width * (2 ** i) // 4
            
            # Residual block
            layers.append(self._make_residual_block(in_ch, out_ch))
            
            # Downsample (except last layer)
            if i < depth - 1:
                layers.append(nn.MaxPool2d(2))
            
            in_ch = out_ch
        
        return nn.ModuleList(layers)
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create residual block."""
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                
                # Skip connection
                self.skip = nn.Identity()
                if in_channels != out_channels:
                    self.skip = nn.Conv2d(in_channels, out_channels, 1)
            
            def forward(self, x):
                identity = self.skip(x)
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                out += identity
                out = self.relu(out)
                
                return out
        
        return ResidualBlock(in_channels, out_channels)
    
    def _get_backbone_channels(self) -> List[int]:
        """Get number of channels at each backbone level."""
        width = self.backbone_cfg['width']
        depth = self.backbone_cfg['depth']
        
        channels = [width // 4]  # Initial layer
        
        for i in range(depth):
            channels.append(width * (2 ** i) // 4)
        
        return channels
    
    def _build_fpn(self) -> nn.Module:
        """Build Feature Pyramid Network."""
        # Simple top-down FPN
        return nn.ModuleDict({
            'lateral': nn.ModuleList([
                nn.Conv2d(ch, self.backbone_cfg['width'], 1)
                for ch in self.backbone_channels[1:]  # Skip first layer
            ]),
            'smooth': nn.ModuleList([
                nn.Conv2d(self.backbone_cfg['width'], self.backbone_cfg['width'], 3, padding=1)
                for _ in self.backbone_channels[1:]
            ])
        })
    
    def forward(self, ir: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through PlumeDetXL.
        
        Args:
            ir: IR video tensor [B, 1, T, H, W]
            
        Returns:
            Dictionary containing:
                - tracks: List of Track objects per batch item
                - zt: Global embedding [B, 256]
                - events: List of EventToken objects
        """
        B, C, T, H, W = ir.shape
        
        # Process each frame independently
        all_detections = []
        frame_features = []
        
        for t in range(T):
            frame = ir[:, :, t]  # [B, 1, H, W]
            
            # Extract features
            features = self._extract_features(frame)  # [B, D, H', W']
            frame_features.append(features)
            
            # Detect objects in frame
            detections = self._detect_frame(features, t)
            all_detections.extend(detections)
        
        # Associate detections into tracks
        tracks_per_batch = self._associate_tracks(all_detections, B, T)
        
        # Global embedding from averaged features
        if frame_features:
            avg_features = torch.stack(frame_features, dim=2).mean(dim=2)  # [B, D, H', W']
            pooled = self.global_pool(avg_features)  # [B, D, 1, 1]
            zt = self.embedding_proj(pooled.view(B, -1))  # [B, pool_to_embed]
        else:
            zt = torch.zeros(B, self.pool_to_embed, device=ir.device)
        
        # Extract events from tracks
        events = self._extract_events(tracks_per_batch)
        
        return {
            'tracks': tracks_per_batch,
            'zt': zt,
            'events': events
        }
    
    def _extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Extract features from single frame.
        
        Args:
            frame: Single frame [B, 1, H, W]
            
        Returns:
            Features [B, D, H', W']
        """
        x = frame
        backbone_features = []
        
        # Forward through backbone
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, tuple):
                # Residual block
                conv_layers, skip = layer
                identity = x
                x = conv_layers(x)
                if skip is not None:
                    identity = skip(identity)
                x = x + identity
                x = F.relu(x, inplace=True)
            else:
                x = layer(x)
            
            # Store features for FPN
            if i > 0 and self.fpn is not None:  # Skip initial conv/pool
                backbone_features.append(x)
        
        # Apply FPN if enabled
        if self.fpn is not None and backbone_features:
            # Top-down pathway
            fpn_features = []
            x = backbone_features[-1]  # Start from top level
            
            for i in range(len(backbone_features) - 1, -1, -1):
                # Lateral connection - adjust index for FPN which skips first layer
                lateral_idx = i  # Use direct index since we collected features from i > 0
                lateral = self.fpn['lateral'][lateral_idx](backbone_features[i])
                
                # Top-down
                if fpn_features:
                    # Upsample previous level
                    upsampled = F.interpolate(
                        fpn_features[-1],
                        size=lateral.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    x = lateral + upsampled
                else:
                    x = lateral
                
                # Smooth
                x = self.fpn['smooth'][lateral_idx](x)
                fpn_features.append(x)
            
            # Use middle level features
            x = fpn_features[len(fpn_features) // 2]
        
        return x
    
    def _detect_frame(self, features: torch.Tensor, t_idx: int) -> List[Dict[str, Any]]:
        """
        Detect objects in single frame.
        
        Args:
            features: Frame features [B, D, H', W']
            t_idx: Frame index
            
        Returns:
            List of detection dictionaries
        """
        B, D, H_f, W_f = features.shape
        
        # Apply detection heads
        heatmap = self.heatmap_head(features)  # [B, 1, H', W']
        offset = self.offset_head(features)    # [B, 2, H', W']
        size = self.size_head(features)        # [B, 2, H', W']
        intensity = self.intensity_head(features)  # [B, 1, H', W']
        
        detections = []
        
        for b in range(B):
            # Find peaks in heatmap
            batch_heatmap = heatmap[b, 0]  # [H', W']
            batch_offset = offset[b]       # [2, H', W']
            batch_size = size[b]           # [2, H', W']
            batch_intensity = intensity[b, 0]  # [H', W']
            
            # Find local maxima above threshold
            peaks = self._find_peaks_2d(
                batch_heatmap,
                threshold=self.heads_cfg['heatmap_thr']
            )
            
            # Limit number of detections
            max_dets = self.heads_cfg['max_dets_per_frame']
            if len(peaks) > max_dets:
                # Keep top-k by score
                peak_scores = [batch_heatmap[y, x].item() for x, y in peaks]
                sorted_indices = sorted(range(len(peaks)), key=lambda i: peak_scores[i], reverse=True)
                peaks = [peaks[i] for i in sorted_indices[:max_dets]]
            
            # Convert peaks to detections
            for peak_x, peak_y in peaks:
                # Get values at peak
                score = batch_heatmap[peak_y, peak_x].item()
                dx = batch_offset[0, peak_y, peak_x].item()
                dy = batch_offset[1, peak_y, peak_x].item()
                w = batch_size[0, peak_y, peak_x].item()
                h = batch_size[1, peak_y, peak_x].item()
                intens = batch_intensity[peak_y, peak_x].item()
                
                # Convert to full resolution coordinates (assuming 4x downsampling)
                scale_factor = 4.0
                center_x = (peak_x + dx) * scale_factor
                center_y = (peak_y + dy) * scale_factor
                width = w * scale_factor
                height = h * scale_factor
                
                detection = {
                    'batch_idx': b,
                    't_idx': t_idx,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'score': score,
                    'intensity': intens
                }
                
                detections.append(detection)
        
        return detections
    
    def _find_peaks_2d(
        self,
        heatmap: torch.Tensor,
        threshold: float,
        kernel_size: int = 3
    ) -> List[Tuple[int, int]]:
        """
        Find 2D peaks in heatmap.
        
        Args:
            heatmap: 2D heatmap tensor [H, W]
            threshold: Minimum peak value
            kernel_size: Size of local maximum kernel
            
        Returns:
            List of (x, y) peak coordinates
        """
        # Apply local maximum pooling
        padding = kernel_size // 2
        local_max = F.max_pool2d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).squeeze()
        
        # Find pixels that are local maxima and above threshold
        is_peak = (heatmap == local_max) & (heatmap >= threshold)
        
        # Get coordinates
        peak_coords = torch.nonzero(is_peak, as_tuple=False)  # [N, 2] as (y, x)
        
        # Convert to (x, y) format
        peaks = [(int(coord[1]), int(coord[0])) for coord in peak_coords]
        
        return peaks
    
    def _associate_tracks(
        self,
        detections: List[Dict[str, Any]],
        batch_size: int,
        num_frames: int
    ) -> List[List[Track]]:
        """
        Associate detections into tracks using nearest neighbor.
        
        Args:
            detections: List of all detections
            batch_size: Number of batch items
            num_frames: Number of frames
            
        Returns:
            List of track lists (one per batch item)
        """
        tracks_per_batch = [[] for _ in range(batch_size)]
        
        # Group detections by batch
        batch_detections = [[] for _ in range(batch_size)]
        for det in detections:
            batch_detections[det['batch_idx']].append(det)
        
        # Process each batch independently
        for b in range(batch_size):
            batch_tracks = []
            dets = batch_detections[b]
            
            # Sort detections by frame
            dets.sort(key=lambda x: x['t_idx'])
            
            # Simple nearest neighbor tracking
            for det in dets:
                # Find closest existing track
                best_track = None
                best_distance = float('inf')
                
                for track in batch_tracks:
                    if not track.active:
                        continue
                    
                    # Check frame gap constraint
                    if det['t_idx'] - track.last_update > self.tracker_cfg['max_frame_gap']:
                        continue
                    
                    # Compute distance to track
                    if track.center_xy:
                        last_x, last_y = track.center_xy[-1]
                        distance = math.sqrt(
                            (det['center_x'] - last_x) ** 2 + 
                            (det['center_y'] - last_y) ** 2
                        )
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_track = track
                
                # Associate with track or create new one
                if best_track is not None and best_distance < 50.0:  # Threshold
                    # Add to existing track
                    best_track.add_detection(
                        det['t_idx'],
                        det['center_x'],
                        det['center_y'], 
                        det['width'],
                        det['height'],
                        det['score'],
                        det['intensity']
                    )
                else:
                    # Create new track
                    if len(batch_tracks) < self.tracker_cfg['max_tracks']:
                        track = Track(self.next_track_id)
                        self.next_track_id += 1
                        
                        track.add_detection(
                            det['t_idx'],
                            det['center_x'],
                            det['center_y'],
                            det['width'], 
                            det['height'],
                            det['score'],
                            det['intensity']
                        )
                        
                        batch_tracks.append(track)
            
            tracks_per_batch[b] = batch_tracks
        
        return tracks_per_batch
    
    def _extract_events(self, tracks_per_batch: List[List[Track]]) -> List[EventToken]:
        """
        Extract events from track patterns.
        
        Args:
            tracks_per_batch: Tracks for each batch item
            
        Returns:
            List of EventToken objects
        """
        events = []
        
        for b, tracks in enumerate(tracks_per_batch):
            for track in tracks:
                if len(track.score) == 0:
                    continue
                
                # Event for track appearance
                max_score = max(track.score)
                max_idx = track.score.index(max_score)
                
                if max_score >= 0.5:  # Threshold
                    # Convert frame to milliseconds (assuming 24 FPS)
                    t_ms = int(track.t_idx[max_idx] * (1000 / 24))
                    
                    event = EventToken(
                        type="ir_hotspot",
                        score=max_score,
                        t_ms=t_ms,
                        meta={
                            'track_id': track.id,
                            'max_intensity': max(track.intensity) if track.intensity else 0.0,
                            'duration_frames': len(track.t_idx),
                            'center_xy': track.center_xy[max_idx] if track.center_xy else (0, 0),
                            'batch_idx': b
                        }
                    )
                    
                    events.append(event)
        
        return events
    
    def get_output_shapes(self, input_shape: Tuple[int, ...]) -> Dict[str, Tuple[int, ...]]:
        """
        Get expected output shapes for given input shape.
        
        Args:
            input_shape: Input tensor shape (B, C, T, H, W)
            
        Returns:
            Dictionary of output shapes
        """
        B, C, T, H, W = input_shape
        
        return {
            'tracks': "List[List[Track]]",
            'zt': (B, self.pool_to_embed),
            'events': "List[EventToken]"
        }