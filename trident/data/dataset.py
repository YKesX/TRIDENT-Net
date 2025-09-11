"""
Dataset and data loading utilities for TRIDENT-Net.

Author: Yağızhan Keskin
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .video_ring import VideoRing
from .transforms import AlbuStereoClip


class VideoJsonlDataset(Dataset):
    """
    Dataset for video clips loaded from JSONL metadata.
    
    Supports variable-T video clips with synchronized RGB/IR streams,
    kinematics data, and optional class embeddings. Handles missing
    modalities gracefully.
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train"
    ) -> None:
        """
        Initialize dataset from configuration.
        
        Args:
            cfg: Configuration from tasks.yml.data.dataset section
            split: Dataset split ("train", "val", "test")
        """
        self.cfg = cfg
        self.split = split
        
        # Load data sources configuration
        sources_cfg = cfg.get('sources', {})
        self.jsonl_path = sources_cfg.get('jsonl_path')
        self.video_root = Path(sources_cfg.get('video_root', '.'))
        
        # Field mapping configuration
        fields_map = cfg.get('fields_map', {})
        self.video_path_key = fields_map.get('video_path_key', 'video.path')
        self.rgb_path_key = fields_map.get('rgb_path_key', 'video.rgb_path')
        self.ir_path_key = fields_map.get('ir_path_key', 'video.ir_path')
        self.kinematics_key = fields_map.get('kinematics_key', 'radar.kinematics')
        self.prompt_key = fields_map.get('prompt_key', 'prompt')
        self.target_box_key = fields_map.get('target_box_key', 'target.bbox')
        
        # Label keys from tasks.yml.labels.fields
        self.shoot_ms_key = 'shoot_ms'
        self.hit_ms_key = 'hit_ms'
        self.kill_ms_key = 'kill_ms'
        
        # Class ID configuration
        self.class_id_key = 'target.class_id'
        self.class_conf_key = 'target.class_conf'
        
        # Load JSONL data
        self.samples = self._load_jsonl_samples()
        
        # Initialize transforms
        transform_cfg = cfg.get('transforms', {})
        if 'class' in transform_cfg:
            self.transform = AlbuStereoClip(transform_cfg)
        else:
            self.transform = None
        
        # Video processing parameters
        self.fps_assumed = cfg.get('fps_assumed', 24)
        
        # Temporal windows from tasks.yml.preprocess.temporal_windows_ms
        temporal_cfg = cfg.get('temporal_windows_ms', {})
        self.pre_ms = temporal_cfg.get('pre_ms', 1500)
        self.fire_ms = temporal_cfg.get('fire_ms', 600) 
        self.post_ms = temporal_cfg.get('post_ms', 2100)
        
        # Target image size from tasks.yml.preprocess.image_size
        self.target_h = cfg.get('image_size', {}).get('h', 704)
        self.target_w = cfg.get('image_size', {}).get('w', 1248)
        
    def _load_jsonl_samples(self) -> List[Dict[str, Any]]:
        """Load and parse JSONL file."""
        if not self.jsonl_path or not os.path.exists(self.jsonl_path):
            return []  # Return empty for missing file
            
        samples = []
        with open(self.jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num + 1}: {e}")
                    continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample at given index."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.samples)}")
        
        sample_meta = self.samples[idx]
        
        # Load video data
        rgb, ir = self._load_video_streams(sample_meta)
        
        # Load kinematics
        kinematics = self._load_kinematics(sample_meta)
        
        # Load class ID if available
        class_id = self._load_class_id(sample_meta)
        
        # Derive labels from time stamps
        labels = self._derive_labels(sample_meta)
        
        # Extract timing metadata
        times_ms = self._extract_times(sample_meta)
        
        # Apply transforms if available
        if self.transform is not None and rgb is not None:
            rgb, ir, transform_meta = self.transform(rgb, ir)
        else:
            transform_meta = {}
        
        # Convert to tensors and add batch dimension
        result = {
            'rgb': self._to_tensor_with_batch(rgb) if rgb is not None else torch.zeros(1, 3, 10, self.target_h, self.target_w),
            'ir': self._to_tensor_with_batch(ir) if ir is not None else torch.zeros(1, 1, 10, self.target_h, self.target_w),
            'kin': self._to_tensor_with_batch(kinematics),
            'labels': {
                'hit': torch.tensor([[labels['hit']]], dtype=torch.float32),
                'kill': torch.tensor([[labels['kill']]], dtype=torch.float32)
            },
            'meta': {
                'path': self._get_video_path(sample_meta),
                'times_ms': times_ms,
                'prompt': self._get_nested_value(sample_meta, self.prompt_key, ''),
                'transform_meta': transform_meta,
                'sample_idx': idx
            }
        }
        
        # Add class_id if available
        if class_id is not None:
            result['class_id'] = torch.tensor([class_id], dtype=torch.long)
        
        return result
    
    def _load_video_streams(self, sample_meta: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load RGB and IR video streams and slice temporal windows.
        
        Returns:
            tuple: (rgb_clip, ir_clip) as numpy arrays or None
                - RGB: [T, H, W, 3] if available
                - IR: [T, H, W, 1] if available
        """
        # Get video paths
        rgb_path = self._get_video_path(sample_meta, 'rgb')
        ir_path = self._get_video_path(sample_meta, 'ir')
        
        # Load RGB stream
        rgb_clip = None
        if rgb_path and os.path.exists(rgb_path):
            rgb_clip = self._load_video_clip(rgb_path)
        
        # Load IR stream
        ir_clip = None
        if ir_path and os.path.exists(ir_path):
            ir_clip = self._load_video_clip(ir_path)
        
        # If no RGB but have general video path, try loading as RGB
        if rgb_clip is None:
            general_path = self._get_video_path(sample_meta, 'general')
            if general_path and os.path.exists(general_path):
                rgb_clip = self._load_video_clip(general_path)
        
        # Slice temporal windows if timing data available
        times_ms = self._extract_times(sample_meta)
        if times_ms and rgb_clip is not None:
            rgb_clip = self._slice_temporal_windows(rgb_clip, times_ms)
        if times_ms and ir_clip is not None:
            ir_clip = self._slice_temporal_windows(ir_clip, times_ms)
        
        # Letterbox to target size
        if rgb_clip is not None:
            rgb_clip = self._letterbox_clip(rgb_clip, target_shape=(self.target_h, self.target_w, 3))
        if ir_clip is not None:
            ir_clip = self._letterbox_clip(ir_clip, target_shape=(self.target_h, self.target_w, 1))
            
        return rgb_clip, ir_clip
    
    def _get_video_path(self, sample_meta: Dict[str, Any], stream_type: str = 'general') -> Optional[str]:
        """Get video file path for specified stream type."""
        if stream_type == 'rgb':
            path = self._get_nested_value(sample_meta, self.rgb_path_key)
        elif stream_type == 'ir':
            path = self._get_nested_value(sample_meta, self.ir_path_key)
        else:
            path = self._get_nested_value(sample_meta, self.video_path_key)
        
        if not path:
            return None
        
        # Convert to absolute path if relative
        if not os.path.isabs(path):
            path = self.video_root / path
            
        return str(path)
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def _load_video_clip(self, video_path: str) -> Optional[np.ndarray]:
        """
        Load video file and return frames as numpy array.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Clip array [T, H, W, C] or None if loading fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
                
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for RGB streams
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append(frame)
            
            cap.release()
            
            if frames:
                return np.stack(frames, axis=0)  # [T, H, W, C]
            return None
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def _slice_temporal_windows(
        self,
        clip: np.ndarray,
        times_ms: Dict[str, Any]
    ) -> np.ndarray:
        """
        Slice clip to temporal windows around key events.
        
        Args:
            clip: Video clip [T, H, W, C]
            times_ms: Dictionary with timing information
            
        Returns:
            Sliced clip covering pre/fire/post windows
        """
        T = clip.shape[0]
        
        # Get video FPS (assume from total frames and duration if available)
        fps = self.fps_assumed  # Default fallback
        
        # Use VideoRing to compute temporal slicing
        video_ring = VideoRing(fps_hint=fps)
        
        # Compute frame indices for windows
        pre_indices, fire_indices, post_indices = video_ring.freeze_and_slice(
            self.pre_ms, self.fire_ms, self.post_ms, fps
        )
        
        # Combine all indices (adjust for actual video length)
        all_indices = pre_indices + fire_indices + post_indices
        
        # Map negative indices to beginning of clip and positive to end
        shoot_frame = times_ms.get('shoot_ms', 0) * fps // 1000 if 'shoot_ms' in times_ms else T // 2
        shoot_frame = max(0, min(shoot_frame, T - 1))
        
        # Adjust indices relative to shoot frame
        adjusted_indices = []
        for idx in all_indices:
            abs_idx = shoot_frame + idx
            abs_idx = max(0, min(abs_idx, T - 1))  # Clamp to valid range
            adjusted_indices.append(abs_idx)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in adjusted_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        
        # Extract frames
        if unique_indices:
            return clip[unique_indices]
        else:
            return clip  # Return original if slicing fails
    
    def _letterbox_clip(
        self,
        clip: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Letterbox entire clip to target shape.
        
        Args:
            clip: Video clip [T, H, W, C]
            target_shape: Target (H, W, C)
            
        Returns:
            Letterboxed clip [T, target_H, target_W, C]
        """
        T, H, W, C = clip.shape
        target_H, target_W, target_C = target_shape
        
        # Ensure channel count matches
        if C != target_C:
            if C == 3 and target_C == 1:
                # Convert RGB to grayscale
                clip = np.mean(clip, axis=-1, keepdims=True)
            elif C == 1 and target_C == 3:
                # Convert grayscale to RGB
                clip = np.repeat(clip, 3, axis=-1)
        
        # Letterbox each frame
        letterboxed_frames = []
        
        for t in range(T):
            frame = clip[t]  # [H, W, C]
            
            # Calculate scale to fit within target while preserving aspect ratio
            scale = min(target_H / H, target_W / W)
            
            # Calculate new size
            new_H = int(H * scale)
            new_W = int(W * scale)
            
            # Resize frame
            if C == 1:
                resized = cv2.resize(frame.squeeze(-1), (new_W, new_H))
                resized = np.expand_dims(resized, -1)
            else:
                resized = cv2.resize(frame, (new_W, new_H))
            
            # Create letterboxed frame
            letterboxed = np.zeros((target_H, target_W, target_C), dtype=frame.dtype)
            
            # Calculate padding offsets (center the image)
            offset_y = (target_H - new_H) // 2
            offset_x = (target_W - new_W) // 2
            
            # Place resized frame in center
            letterboxed[offset_y:offset_y + new_H, offset_x:offset_x + new_W] = resized
            
            letterboxed_frames.append(letterboxed)
        
        return np.stack(letterboxed_frames, axis=0)
    
    def _load_kinematics(self, sample_meta: Dict[str, Any]) -> np.ndarray:
        """
        Load kinematics data for (pre, fire, post) windows.
        
        Args:
            sample_meta: Sample metadata
            
        Returns:
            Kinematics tensor [3, 9] representing (pre, fire, post) x 9 features
        """
        kin_data = self._get_nested_value(sample_meta, self.kinematics_key)
        
        if kin_data is None:
            # Return default kinematics if not available
            return np.zeros((3, 9), dtype=np.float32)
        
        # Convert to numpy array
        if isinstance(kin_data, list):
            kin_data = np.array(kin_data, dtype=np.float32)
        elif isinstance(kin_data, dict):
            # Extract numeric values if dict format
            values = []
            for key in sorted(kin_data.keys()):
                val = kin_data[key]
                if isinstance(val, (list, tuple)):
                    values.extend(val)
                else:
                    values.append(val)
            kin_data = np.array(values[:27], dtype=np.float32)  # Limit to 27 values
        
        # Reshape to [3, 9] if needed
        if kin_data.ndim == 1:
            if len(kin_data) >= 27:
                kin_data = kin_data[:27].reshape(3, 9)
            else:
                # Pad with zeros if insufficient data
                padded = np.zeros(27, dtype=np.float32)
                padded[:len(kin_data)] = kin_data
                kin_data = padded.reshape(3, 9)
        elif kin_data.shape != (3, 9):
            # Resize/pad to correct shape
            kin_reshaped = np.zeros((3, 9), dtype=np.float32)
            min_rows = min(kin_data.shape[0], 3)
            min_cols = min(kin_data.shape[1], 9) if kin_data.ndim > 1 else min(kin_data.shape[0], 27)
            
            if kin_data.ndim == 1:
                kin_reshaped.flat[:min_cols] = kin_data[:min_cols]
            else:
                kin_reshaped[:min_rows, :min_cols] = kin_data[:min_rows, :min_cols]
            kin_data = kin_reshaped
            
        return kin_data
    
    def _load_class_id(self, sample_meta: Dict[str, Any]) -> Optional[int]:
        """Load class ID if available."""
        class_id = self._get_nested_value(sample_meta, self.class_id_key)
        
        if class_id is not None:
            try:
                return int(class_id)
            except (ValueError, TypeError):
                return None
        
        return None
    
    def _derive_labels(self, sample_meta: Dict[str, Any]) -> Dict[str, float]:
        """
        Derive binary labels from timing information.
        
        Follows hierarchy: kill ⊆ hit ⊆ shoot
        
        Args:
            sample_meta: Sample metadata
            
        Returns:
            Dictionary with hit and kill labels (0.0 or 1.0)
        """
        # Extract timing data
        shoot_ms = self._get_nested_value(sample_meta, self.shoot_ms_key)
        hit_ms = self._get_nested_value(sample_meta, self.hit_ms_key)
        kill_ms = self._get_nested_value(sample_meta, self.kill_ms_key)
        
        # Derive binary labels
        # shoot = 1.0 if shoot_ms is not None else 0.0  # Not used in output
        hit = 1.0 if hit_ms is not None else 0.0
        kill = 1.0 if kill_ms is not None else 0.0
        
        # Enforce hierarchy: if kill=1 then hit=1
        if kill == 1.0:
            hit = 1.0
        
        return {
            'hit': hit,
            'kill': kill
        }
    
    def _extract_times(self, sample_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timing information from sample metadata."""
        return {
            'shoot_ms': self._get_nested_value(sample_meta, self.shoot_ms_key),
            'hit_ms': self._get_nested_value(sample_meta, self.hit_ms_key),
            'kill_ms': self._get_nested_value(sample_meta, self.kill_ms_key)
        }
    
    def _to_tensor_with_batch(self, data: Optional[np.ndarray]) -> torch.Tensor:
        """Convert numpy array to tensor and add batch dimension."""
        if data is None:
            return torch.zeros(1, 1)  # Dummy tensor
        
        tensor = torch.from_numpy(data).float()
        
        # Reshape based on expected format
        if data.ndim == 4:  # Video: [T, H, W, C] -> [B=1, C, T, H, W]
            tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
        elif data.ndim == 2:  # Kinematics: [3, 9] -> [B=1, 3, 9]
            tensor = tensor.unsqueeze(0)  # [1, 3, 9]
        else:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
        return tensor


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal sensor data.
    
    Expected data structure:
    data_root/
        rgb_roi/
        rgb_roi_t/
        rgb_pre/
        rgb_post/
        ir_roi_t/
        rd_seq/
        pulse_feat/
        rd_tokens/
        labels.npy
        metadata.npy
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        transform: Optional[callable] = None,
        load_all_modalities: bool = True,
    ) -> None:
        """
        Args:
            data_root: Root directory containing data
            split: Data split ('train', 'val', 'test')
            transform: Optional data transformation function
            load_all_modalities: Whether to load all modalities or subset
        """
        self.data_root = Path(data_root) / split
        self.transform = transform
        self.load_all_modalities = load_all_modalities
        
        # Load metadata and labels
        self.labels = np.load(self.data_root / "labels.npy")
        self.metadata = np.load(self.data_root / "metadata.npy", allow_pickle=True)
        
        self.length = len(self.labels)
        
        # Check which modalities are available
        self.available_modalities = self._check_available_modalities()
    
    def _check_available_modalities(self) -> Dict[str, bool]:
        """Check which modality directories exist."""
        modalities = {
            "rgb_roi": "rgb_roi",
            "rgb_roi_t": "rgb_roi_t", 
            "rgb_pre": "rgb_pre",
            "rgb_post": "rgb_post",
            "ir_roi_t": "ir_roi_t",
            "rd_seq": "rd_seq",
            "pulse_feat": "pulse_feat",
            "rd_tokens": "rd_tokens",
        }
        
        available = {}
        for key, dirname in modalities.items():
            available[key] = (self.data_root / dirname).exists()
        
        return available
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample at given index."""
        sample = {}
        
        # Load labels and metadata
        sample["y_outcome"] = self.labels[idx]
        sample["meta"] = self.metadata[idx] if len(self.metadata) > idx else {}
        
        # Load available modalities
        if self.available_modalities.get("rgb_roi", False):
            sample["rgb_roi"] = self._load_tensor(f"rgb_roi/{idx:06d}.npy")
            
        if self.available_modalities.get("rgb_roi_t", False):
            sample["rgb_roi_t"] = self._load_tensor(f"rgb_roi_t/{idx:06d}.npy")
            
        if self.available_modalities.get("rgb_pre", False):
            sample["rgb_pre"] = self._load_tensor(f"rgb_pre/{idx:06d}.npy")
            
        if self.available_modalities.get("rgb_post", False):
            sample["rgb_post"] = self._load_tensor(f"rgb_post/{idx:06d}.npy")
            
        if self.available_modalities.get("ir_roi_t", False):
            sample["ir_roi_t"] = self._load_tensor(f"ir_roi_t/{idx:06d}.npy")
            
        if self.available_modalities.get("rd_seq", False):
            sample["rd_seq"] = self._load_tensor(f"rd_seq/{idx:06d}.npy")
            
        if self.available_modalities.get("pulse_feat", False):
            sample["pulse_feat"] = self._load_tensor(f"pulse_feat/{idx:06d}.npy")
            
        if self.available_modalities.get("rd_tokens", False):
            sample["rd_tokens"] = self._load_tensor(f"rd_tokens/{idx:06d}.npy")
        
        # Add geometric and prior information from metadata
        meta = sample["meta"]
        sample["geom"] = meta.get("geometry", {})
        sample["priors"] = meta.get("priors", {})
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_tensor(self, rel_path: str) -> torch.Tensor:
        """Load numpy array and convert to tensor."""
        try:
            data = np.load(self.data_root / rel_path)
            return torch.from_numpy(data).float()
        except FileNotFoundError:
            # Return dummy tensor for missing files
            return torch.zeros(1)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a sample without loading data."""
        meta = self.metadata[idx] if len(self.metadata) > idx else {}
        return {
            "index": idx,
            "label": self.labels[idx],
            "metadata": meta,
            "available_modalities": [k for k, v in self.available_modalities.items() if v],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for multimodal batches.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors
    """
    if not batch:
        return {}
    
    # Get all keys from first sample
    keys = set(batch[0].keys())
    
    # Separate tensor and non-tensor keys
    tensor_keys = []
    other_keys = []
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            tensor_keys.append(key)
        else:
            other_keys.append(key)
    
    result = {}
    
    # Stack tensor data
    for key in tensor_keys:
        tensors = [sample[key] for sample in batch if key in sample]
        if tensors:
            try:
                result[key] = torch.stack(tensors, dim=0)
            except RuntimeError:
                # Handle variable size tensors by padding
                result[key] = _pad_and_stack(tensors)
    
    # Collect non-tensor data
    for key in other_keys:
        result[key] = [sample.get(key, None) for sample in batch]
    
    return result


def _pad_and_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad tensors to same size and stack."""
    if not tensors:
        return torch.empty(0)
    
    # Find maximum dimensions
    max_dims = []
    for dim in range(tensors[0].dim()):
        max_size = max(t.shape[dim] for t in tensors)
        max_dims.append(max_size)
    
    # Pad tensors
    padded = []
    for tensor in tensors:
        pad_widths = []
        for dim in range(tensor.dim()):
            pad_width = max_dims[dim] - tensor.shape[dim]
            pad_widths.extend([0, pad_width])
        
        # Reverse padding for F.pad (needs last-dim-first order)
        pad_widths = pad_widths[::-1]
        padded_tensor = torch.nn.functional.pad(tensor, pad_widths)
        padded.append(padded_tensor)
    
    return torch.stack(padded, dim=0)


class MultimodalSubset(Dataset):
    """Subset of multimodal dataset with specific modalities."""
    
    def __init__(
        self,
        dataset: MultimodalDataset,
        modalities: List[str],
        indices: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            dataset: Parent multimodal dataset
            modalities: List of modalities to include
            indices: Optional subset of indices to use
        """
        self.dataset = dataset
        self.modalities = set(modalities)
        self.indices = indices if indices is not None else list(range(len(dataset)))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get filtered sample."""
        actual_idx = self.indices[idx]
        full_sample = self.dataset[actual_idx]
        
        # Filter to requested modalities + always include labels and metadata
        keep_keys = self.modalities | {"y_outcome", "meta", "geom", "priors"}
        filtered_sample = {k: v for k, v in full_sample.items() if k in keep_keys}
        
        return filtered_sample


def create_data_loaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[callable] = None,
    val_transform: Optional[callable] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test data loaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        transform = train_transform if split == "train" else val_transform
        try:
            datasets[split] = MultimodalDataset(
                data_root=data_root,
                split=split,
                transform=transform,
            )
        except FileNotFoundError:
            datasets[split] = None
    
    loaders = []
    for split in ["train", "val", "test"]:
        if datasets[split] is not None:
            shuffle = (split == "train")
            loader = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
            )
            loaders.append(loader)
        else:
            loaders.append(None)
    
    return tuple(loaders)