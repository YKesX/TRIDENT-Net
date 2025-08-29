"""
Synthetic data generation for testing and development.

Generates realistic-looking multimodal data for system validation.
"""

from typing import Dict, Any, Optional

import torch
import numpy as np


class SyntheticDataGenerator:
    """Generator for synthetic multimodal data."""
    
    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        sequence_length: int = 8,
        radar_features: int = 64,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize synthetic data generator.
        
        Args:
            image_size: Height and width for generated images
            sequence_length: Default sequence length for temporal data
            radar_features: Number of radar feature dimensions
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.radar_features = radar_features
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def generate_sample(
        self,
        load_rgb: bool = True,
        load_ir: bool = True,
        load_radar: bool = True,
        sequence_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic sample.
        
        Args:
            load_rgb: Generate RGB data
            load_ir: Generate IR data
            load_radar: Generate radar data
            sequence_length: Override default sequence length
            
        Returns:
            Dictionary with synthetic multimodal data
        """
        seq_len = sequence_length or self.sequence_length
        sample = {}
        
        # Generate RGB data
        if load_rgb:
            sample.update(self._generate_rgb_data(seq_len))
        
        # Generate IR/thermal data
        if load_ir:
            sample.update(self._generate_ir_data(seq_len))
        
        # Generate radar data
        if load_radar:
            sample.update(self._generate_radar_data(seq_len))
        
        # Generate labels and metadata
        sample.update(self._generate_labels_and_metadata())
        
        return sample
    
    def _generate_rgb_data(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic RGB data."""
        h, w = self.image_size
        
        # RGB ROI (single frame)
        rgb_roi = self._generate_image_with_objects(h, w, num_objects=2)
        
        # RGB temporal sequence
        rgb_seq = torch.stack([
            self._generate_image_with_objects(h, w, num_objects=1 + i % 3)
            for i in range(seq_len)
        ])
        
        # RGB pre/post for change detection
        base_scene = self._generate_background(h, w)
        rgb_pre = base_scene + 0.1 * torch.randn_like(base_scene)
        
        # Add some changes for post image
        change_mask = self._generate_change_mask(h, w)
        rgb_post = rgb_pre.clone()
        rgb_post[change_mask] += 0.3 + 0.2 * torch.randn(change_mask.sum())
        
        return {
            "rgb_roi": rgb_roi,
            "rgb_roi_t": rgb_seq,
            "rgb_pre": rgb_pre,
            "rgb_post": rgb_post,
        }
    
    def _generate_ir_data(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic IR/thermal data."""
        h, w = self.image_size
        
        # IR temporal sequence (single channel thermal)
        ir_seq = torch.stack([
            self._generate_thermal_image(h, w, hotspots=2 + i % 3)
            for i in range(seq_len)
        ])
        
        return {"ir_roi_t": ir_seq}
    
    def _generate_radar_data(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic radar data."""
        # Radar sequence (micro-doppler)
        rd_seq = self._generate_doppler_sequence(seq_len)
        
        # Pulse features
        pulse_feat = torch.randn(self.radar_features)
        pulse_feat[::4] += 2.0  # Add some structure
        
        # Radar tokens (simplified representation)
        rd_tokens = rd_seq.mean(dim=0, keepdim=True)  # Aggregate features
        
        return {
            "rd_seq": rd_seq,
            "pulse_feat": pulse_feat,
            "rd_tokens": rd_tokens,
        }
    
    def _generate_labels_and_metadata(self) -> Dict[str, Any]:
        """Generate labels and metadata."""
        # Binary outcome (random but weighted)
        y_outcome = float(self.rng.random() > 0.7)  # 30% positive rate
        
        # Geometry metadata
        geom = {
            "range_m": 100.0 + 50.0 * self.rng.random(),
            "azimuth_deg": 360.0 * self.rng.random(),
            "elevation_deg": 45.0 * self.rng.random(),
            "area_m2": 10.0 + 20.0 * self.rng.random(),
        }
        
        # Priors
        priors = {
            "detection_threshold": 0.5,
            "confidence_min": 0.8,
            "time_window_s": 5.0,
        }
        
        return {
            "y_outcome": y_outcome,
            "geom": geom,
            "priors": priors,
        }
    
    def _generate_image_with_objects(
        self, 
        h: int, 
        w: int, 
        num_objects: int = 2
    ) -> torch.Tensor:
        """Generate RGB image with synthetic objects."""
        # Start with background
        image = self._generate_background(h, w)
        
        # Add objects
        for _ in range(num_objects):
            # Random object position and size
            obj_h = int(20 + 30 * self.rng.random())
            obj_w = int(20 + 30 * self.rng.random())
            y = int(self.rng.random() * (h - obj_h))
            x = int(self.rng.random() * (w - obj_w))
            
            # Generate object with some structure
            obj_brightness = 0.3 + 0.4 * self.rng.random()
            object_patch = torch.ones(3, obj_h, obj_w) * obj_brightness
            
            # Add some texture
            texture = 0.1 * torch.randn(3, obj_h, obj_w)
            object_patch += texture
            
            # Add circular mask for more realistic objects
            y_coords, x_coords = torch.meshgrid(
                torch.arange(obj_h, dtype=torch.float32),
                torch.arange(obj_w, dtype=torch.float32),
                indexing="ij"
            )
            center_y, center_x = obj_h // 2, obj_w // 2
            distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            mask = distances <= min(obj_h, obj_w) // 2
            
            # Apply object to image
            image[:, y:y+obj_h, x:x+obj_w][mask.unsqueeze(0).expand(3, -1, -1)] = \
                object_patch[mask.unsqueeze(0).expand(3, -1, -1)]
        
        return torch.clamp(image, 0, 1)
    
    def _generate_background(self, h: int, w: int) -> torch.Tensor:
        """Generate realistic background pattern."""
        # Create smooth background with some structure
        background = torch.zeros(3, h, w)
        
        # Add low-frequency noise
        for scale in [32, 16, 8]:
            noise_h, noise_w = h // scale, w // scale
            noise = torch.randn(3, noise_h, noise_w)
            noise = torch.nn.functional.interpolate(
                noise.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            background += 0.1 * noise
        
        # Base brightness
        background += 0.3
        
        return background
    
    def _generate_thermal_image(
        self, 
        h: int, 
        w: int, 
        hotspots: int = 2
    ) -> torch.Tensor:
        """Generate thermal image with hotspots."""
        # Start with cool background
        thermal = torch.full((1, h, w), 0.2)  # Cool temperature
        
        # Add hotspots
        for _ in range(hotspots):
            # Random hotspot position and size
            spot_size = int(10 + 20 * self.rng.random())
            y = int(self.rng.random() * (h - spot_size))
            x = int(self.rng.random() * (w - spot_size))
            
            # Create Gaussian hotspot
            y_coords, x_coords = torch.meshgrid(
                torch.arange(spot_size, dtype=torch.float32),
                torch.arange(spot_size, dtype=torch.float32),
                indexing="ij"
            )
            center_y, center_x = spot_size // 2, spot_size // 2
            
            # Gaussian intensity
            sigma = spot_size / 4
            intensity = torch.exp(-((y_coords - center_y)**2 + (x_coords - center_x)**2) / (2 * sigma**2))
            intensity *= 0.6 + 0.3 * self.rng.random()  # Random temperature
            
            # Add to thermal image
            thermal[0, y:y+spot_size, x:x+spot_size] += intensity
        
        # Add some noise
        thermal += 0.05 * torch.randn_like(thermal)
        
        return torch.clamp(thermal, 0, 1)
    
    def _generate_change_mask(self, h: int, w: int) -> torch.Tensor:
        """Generate mask for areas of change."""
        # Create some random change regions
        mask = torch.zeros(h, w, dtype=torch.bool)
        
        num_changes = 1 + int(3 * self.rng.random())
        for _ in range(num_changes):
            # Random change region
            change_h = int(20 + 40 * self.rng.random())
            change_w = int(20 + 40 * self.rng.random())
            y = int(self.rng.random() * (h - change_h))
            x = int(self.rng.random() * (w - change_w))
            
            mask[y:y+change_h, x:x+change_w] = True
        
        return mask
    
    def _generate_doppler_sequence(self, seq_len: int) -> torch.Tensor:
        """Generate synthetic micro-doppler sequence."""
        # Create structured radar sequence
        freq_bins = self.radar_features
        sequence = torch.zeros(seq_len, freq_bins)
        
        # Add some moving targets with doppler shifts
        num_targets = 1 + int(2 * self.rng.random())
        
        for target in range(num_targets):
            # Target with some velocity
            base_freq = int(freq_bins * (0.2 + 0.6 * self.rng.random()))
            velocity = 0.1 + 0.3 * self.rng.random()  # Doppler shift rate
            amplitude = 0.5 + 0.5 * self.rng.random()
            
            for t in range(seq_len):
                # Doppler frequency changes over time
                freq_offset = int(velocity * t * freq_bins / seq_len)
                freq_idx = (base_freq + freq_offset) % freq_bins
                
                # Add signal with some spread
                for spread in range(-2, 3):
                    idx = (freq_idx + spread) % freq_bins
                    sequence[t, idx] += amplitude * np.exp(-spread**2 / 2) / np.sqrt(2 * np.pi)
        
        # Add noise
        sequence += 0.1 * torch.randn_like(sequence)
        
        return torch.clamp(sequence, 0, None)


def generate_synthetic_sample(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to generate a single synthetic sample.
    
    Args:
        **kwargs: Arguments passed to SyntheticDataGenerator.generate_sample()
        
    Returns:
        Synthetic sample dictionary
    """
    generator = SyntheticDataGenerator()
    return generator.generate_sample(**kwargs)