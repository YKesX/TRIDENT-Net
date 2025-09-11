"""
Collation functions for variable-length temporal sequences in TRIDENT-Net.

Implements pad_tracks_collate for batching variable-T sequences
as specified in tasks.yml v0.4.1.

Author: Yağızhan Keskin
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_tracks_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for variable-length temporal sequences.
    
    Handles batching of video clips with different temporal lengths by
    padding sequences to the maximum length in the batch.
    
    Args:
        batch: List of sample dictionaries containing:
            - rgb_seq: [3, T_i, 720, 1280]
            - ir_seq: [1, T_i, 720, 1280] 
            - k_seq: [3, 9] (kinematics)
            - y_outcome: dict with hit/kill labels
            - class_id: (optional) class identifier
            
    Returns:
        Batched dictionary with padded sequences:
            - rgb_seq: [B, 3, T_max, 720, 1280]
            - ir_seq: [B, 1, T_max, 720, 1280]
            - k_seq: [B, 3, 9]
            - y_outcome: dict with batched labels
            - lengths: [B] original sequence lengths
            - mask: [B, T_max] padding mask (1=real, 0=padding)
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
    
    batch_size = len(batch)
    
    # Extract sequences and find max temporal length
    rgb_sequences = []
    ir_sequences = []
    k_sequences = []
    outcomes = []
    class_ids = []
    lengths = []
    
    for sample in batch:
        # RGB sequence [3, T, 720, 1280]
        rgb_seq = sample['rgb_seq']
        if rgb_seq.dim() != 4 or rgb_seq.shape[0] != 3:
            raise ValueError(f"Expected rgb_seq shape [3, T, 720, 1280], got {rgb_seq.shape}")
        if rgb_seq.shape[2:] != (720, 1280):
            raise ValueError(f"Expected native resolution 720×1280, got {rgb_seq.shape[2:]}")
        
        # IR sequence [1, T, 720, 1280]  
        ir_seq = sample['ir_seq']
        if ir_seq.dim() != 4 or ir_seq.shape[0] != 1:
            raise ValueError(f"Expected ir_seq shape [1, T, 720, 1280], got {ir_seq.shape}")
        if ir_seq.shape[2:] != (720, 1280):
            raise ValueError(f"Expected native resolution 720×1280, got {ir_seq.shape[2:]}")
        
        # Ensure RGB and IR have same temporal length
        T = rgb_seq.shape[1]
        if ir_seq.shape[1] != T:
            raise ValueError(f"RGB and IR temporal lengths must match: {T} vs {ir_seq.shape[1]}")
        
        # Kinematics [3, 9]
        k_seq = sample['k_seq']
        if k_seq.shape != (3, 9):
            raise ValueError(f"Expected k_seq shape [3, 9], got {k_seq.shape}")
        
        rgb_sequences.append(rgb_seq)
        ir_sequences.append(ir_seq)
        k_sequences.append(k_seq)
        lengths.append(T)
        outcomes.append(sample['y_outcome'])
        
        if 'class_id' in sample:
            class_ids.append(sample['class_id'])
    
    # Find maximum temporal length
    max_T = max(lengths)
    
    # Pad RGB sequences to max length
    padded_rgb = []
    for rgb_seq in rgb_sequences:
        T = rgb_seq.shape[1]
        if T < max_T:
            # Pad by repeating last frame
            last_frame = rgb_seq[:, -1:].expand(-1, max_T - T, -1, -1)
            padded_seq = torch.cat([rgb_seq, last_frame], dim=1)
        else:
            padded_seq = rgb_seq
        padded_rgb.append(padded_seq)
    
    # Pad IR sequences to max length
    padded_ir = []
    for ir_seq in ir_sequences:
        T = ir_seq.shape[1]
        if T < max_T:
            # Pad by repeating last frame
            last_frame = ir_seq[:, -1:].expand(-1, max_T - T, -1, -1)
            padded_seq = torch.cat([ir_seq, last_frame], dim=1)
        else:
            padded_seq = ir_seq
        padded_ir.append(padded_seq)
    
    # Stack sequences into batches
    rgb_batch = torch.stack(padded_rgb, dim=0)  # [B, 3, T_max, 720, 1280]
    ir_batch = torch.stack(padded_ir, dim=0)    # [B, 1, T_max, 720, 1280]
    k_batch = torch.stack(k_sequences, dim=0)   # [B, 3, 9]
    
    # Create padding mask (1=real frame, 0=padding)
    mask = torch.zeros((batch_size, max_T), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    # Collate outcomes
    collated_outcomes = {}
    if outcomes:
        outcome_keys = outcomes[0].keys()
        for key in outcome_keys:
            values = [outcome[key] for outcome in outcomes]
            if isinstance(values[0], torch.Tensor):
                # Stack tensor values
                collated_outcomes[key] = torch.stack(values, dim=0)
            else:
                # Keep as list for non-tensor values
                collated_outcomes[key] = values
    
    # Build final batch
    result = {
        'rgb_seq': rgb_batch,
        'ir_seq': ir_batch,
        'k_seq': k_batch,
        'y_outcome': collated_outcomes,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'mask': mask
    }
    
    # Add class_ids if present
    if class_ids:
        if isinstance(class_ids[0], torch.Tensor):
            result['class_id'] = torch.stack(class_ids, dim=0)
        else:
            result['class_id'] = torch.tensor(class_ids, dtype=torch.long)
    
    return result


def unpad_sequence(
    padded_seq: torch.Tensor,
    lengths: torch.Tensor
) -> List[torch.Tensor]:
    """
    Unpad a batched sequence back to individual sequences.
    
    Args:
        padded_seq: Padded tensor [B, ...] with time dimension
        lengths: Original lengths [B]
        
    Returns:
        List of unpadded sequences
    """
    sequences = []
    for i, length in enumerate(lengths):
        if padded_seq.dim() == 4:  # [B, C, T, H, W]
            seq = padded_seq[i, :, :length]
        elif padded_seq.dim() == 3:  # [B, T, D]
            seq = padded_seq[i, :length]
        else:
            seq = padded_seq[i]
        sequences.append(seq)
    return sequences


def create_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create attention mask for variable-length sequences.
    
    Args:
        lengths: Sequence lengths [B]
        max_length: Maximum sequence length
        
    Returns:
        Attention mask [B, max_length] where 1=attend, 0=ignore
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
    mask = mask < lengths.unsqueeze(1)
    return mask.float()