"""
TRIDENT-R2: GeoMLP - MLP embedding over concatenated kinematics and features

Author: Yağızhan Keskin
"""

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import BranchModule, EventToken


class GeoMLP(BranchModule):
    """
    Learned embedding on concatenated raw & delta kinematics plus r1 features.
    
    Takes augmented kinematic features (raw + deltas + physics features)
    and learns a compact embedding representation.
    
    Input: k_aug (B, 69) - concat[k0,k1,k2, Δk01,Δk12, r_feats] => 27+18+24=69
    Outputs:
        - zr2 (B, 192) - learned embedding
        - events (list) - MLP-detected events
    """
    
    def __init__(self, hidden: List[int] = None, norm: str = "layer", 
                 activation: str = "gelu", dropout: float = 0.05):
        super().__init__(out_dim=192)
        
        if hidden is None:
            hidden = [128, 192]
        
        self.input_dim = 69  # As specified in tasks.yml
        self.hidden = hidden
        self.norm = norm
        self.activation = activation
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for i, h_dim in enumerate(hidden):
            # Linear layer
            layers.append(nn.Linear(prev_dim, h_dim))
            
            # Normalization
            if norm == "layer":
                layers.append(nn.LayerNorm(h_dim))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(h_dim))
            
            # Activation
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "swish":
                layers.append(nn.SiLU())
            
            # Dropout (except for last layer)
            if i < len(hidden) - 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Feature importance analysis (for event detection)
        self.importance_head = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, k_aug: torch.Tensor) -> Tuple[torch.Tensor, List[EventToken]]:
        """
        Forward pass through GeoMLP.
        
        Args:
            k_aug: Augmented kinematics (B, 69)
                   Format: [k0, k1, k2, Δk01, Δk12, r_feats]
                   where k0,k1,k2 are 9D each (27 total)
                         Δk01,Δk12 are 9D each (18 total)  
                         r_feats are 24D physics features
                         
        Returns:
            tuple: (zr2, events)
                - zr2: (B, 192) learned embedding
                - events: List of MLP-detected events
        """
        B, D = k_aug.shape
        assert D == 69, f"Expected k_aug dimension 69, got {D}"
        
        # Forward through MLP
        zr2 = self.mlp(k_aug)  # (B, 192)
        
        # Compute feature importance for interpretability
        feature_importance = self.importance_head(zr2)  # (B, 69)
        
        # Extract MLP-based events
        # events = self._extract_mlp_events(k_aug, zr2, feature_importance)
        events = []  # TODO: Fix EventToken interface
        
        return zr2, events
    
    def _extract_mlp_events(self, k_aug: torch.Tensor, zr2: torch.Tensor, 
                           feature_importance: torch.Tensor) -> List[EventToken]:
        """Extract events based on MLP processing and feature importance."""
        events = []
        B = k_aug.shape[0]
        
        # Parse input features
        k_raw = k_aug[:, :27].view(B, 3, 9)  # (B, 3, 9) - raw kinematics
        k_deltas = k_aug[:, 27:45].view(B, 2, 9)  # (B, 2, 9) - deltas
        r_feats = k_aug[:, 45:]  # (B, 24) - physics features
        
        for b in range(B):
            # Analyze feature importance patterns
            importance = feature_importance[b]  # (69,)
            
            # Split importance by feature type
            raw_importance = importance[:27].view(3, 9)  # (3, 9)
            delta_importance = importance[27:45].view(2, 9)  # (2, 9) 
            physics_importance = importance[45:]  # (24,)
            
            # Event 1: High importance on position features (spatial focus)
            position_importance = raw_importance[:, :3].mean()  # avg over x,y,z
            if position_importance > 0.7:
                event = EventToken(
                    event_type="spatial_focus",
                    confidence=position_importance.item(),
                    location=(0, 0),
                    timestamp=raw_importance[:, :3].mean(dim=1).argmax().item(),
                    source="geomlp",
                    metadata={
                        'position_importance': position_importance.item(),
                        'max_position_time': raw_importance[:, :3].mean(dim=1).argmax().item(),
                        'batch_idx': b
                    }
                )
                events.append(event)
            
            # Event 2: High importance on velocity deltas (acceleration events)
            velocity_delta_importance = delta_importance[:, 3:6].mean()  # avg over vx,vy,vz deltas
            if velocity_delta_importance > 0.6:
                event = EventToken(
                    event_type="velocity_change",
                    confidence=velocity_delta_importance.item(),
                    location=(0, 0),
                    timestamp=delta_importance[:, 3:6].mean(dim=1).argmax().item() + 1,  # +1 for delta timing
                    source="geomlp",
                    metadata={
                        'velocity_delta_importance': velocity_delta_importance.item(),
                        'max_delta_time': delta_importance[:, 3:6].mean(dim=1).argmax().item(),
                        'batch_idx': b
                    }
                )
                events.append(event)
            
            # Event 3: High importance on physics features (physics-driven behavior)
            max_physics_importance = physics_importance.max()
            max_physics_idx = physics_importance.argmax()
            
            if max_physics_importance > 0.8:
                # Map physics feature index to feature name
                physics_feature_names = [
                    'los_norm_t0', 'los_norm_t1', 'los_norm_t2',  # 0-2
                    'range_rate_01', 'range_rate_12',  # 3-4
                    'closing_speed_t0', 'closing_speed_t1', 'closing_speed_t2',  # 5-7
                    'bearing_rate_01', 'bearing_rate_12',  # 8-9
                    'elevation_rate_01', 'elevation_rate_12',  # 10-11
                    'lateral_miss_t0', 'lateral_miss_t1', 'lateral_miss_t2',  # 12-14
                    'accel_x_t0', 'accel_y_t0', 'accel_z_t0',  # 15-17
                    'accel_x_t1', 'accel_y_t1', 'accel_z_t1',  # 18-20
                    'accel_x_t2', 'accel_y_t2', 'accel_z_t2'   # 21-23
                ]
                
                feature_name = physics_feature_names[max_physics_idx.item()] if max_physics_idx < len(physics_feature_names) else f"feature_{max_physics_idx}"
                
                event = EventToken(
                    event_type="physics_anomaly",
                    confidence=max_physics_importance.item(),
                    location=(0, 0),
                    timestamp=2,  # End of sequence for physics analysis
                    source="geomlp",
                    metadata={
                        'physics_feature': feature_name,
                        'physics_importance': max_physics_importance.item(),
                        'feature_index': max_physics_idx.item(),
                        'batch_idx': b
                    }
                )
                events.append(event)
            
            # Event 4: Embedding magnitude analysis (overall significance)
            embedding_magnitude = torch.norm(zr2[b]).item()
            if embedding_magnitude > 10.0:  # High embedding magnitude threshold
                event = EventToken(
                    event_type="high_kinematic_complexity",
                    confidence=min(1.0, embedding_magnitude / 20.0),
                    location=(0, 0),
                    timestamp=1,  # Middle of sequence
                    source="geomlp",
                    metadata={
                        'embedding_magnitude': embedding_magnitude,
                        'mean_feature_importance': importance.mean().item(),
                        'max_feature_importance': importance.max().item(),
                        'batch_idx': b
                    }
                )
                events.append(event)
        
        return events