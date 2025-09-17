"""
Memory-efficient attention implementations using PyTorch SDPA.

Replaces standard attention with F.scaled_dot_product_attention for better memory usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SDPAAttention(nn.Module):
    """
    Memory-efficient attention using PyTorch's scaled_dot_product_attention.
    
    Automatically uses FlashAttention kernels when available.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using PyTorch SDPA.
        
        Args:
            query: Query tensor (B, L, E)
            key: Key tensor (B, S, E) 
            value: Value tensor (B, S, E)
            attn_mask: Attention mask (L, S) or (B, L, S)
            key_padding_mask: Key padding mask (B, S)
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, E = query.shape
        _, S, _ = key.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)  # (B, L, E)
        k = self.k_proj(key)    # (B, S, E)
        v = self.v_proj(value)  # (B, S, E)
        
        # Reshape for multi-head attention: (B, H, L, d)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Broadcast (L, S) -> (B, H, L, S)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                # Broadcast (B, L, S) -> (B, H, L, S)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Handle key padding mask
        if key_padding_mask is not None:
            # Convert key_padding_mask (B, S) to attention mask (B, H, L, S)
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, L, -1)
            if attn_mask is None:
                attn_mask = key_mask
            else:
                attn_mask = attn_mask.masked_fill(key_mask, float('-inf'))
        
        # Apply scaled dot product attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal
        )
        
        # Reshape back: (B, H, L, d) -> (B, L, E)
        out = out.transpose(1, 2).contiguous().view(B, L, E)
        
        # Output projection
        out = self.out_proj(out)
        
        # Return weights if needed (approximate for compatibility)
        attn_weights = None
        if need_weights:
            # Compute approximate attention weights for backward compatibility
            # Note: This is less memory efficient but sometimes needed
            with torch.no_grad():
                scale = 1.0 / (self.head_dim ** 0.5)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_weights = attn_weights.mean(dim=1)  # Average over heads
        
        return out, attn_weights


class SDPASelfAttention(SDPAAttention):
    """Self-attention using SDPA."""
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return super().forward(x, x, x, attn_mask, key_padding_mask, need_weights)


class SDPACrossAttention(SDPAAttention):
    """Cross-attention using SDPA."""
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return super().forward(query, key_value, key_value, attn_mask, key_padding_mask, need_weights)


def convert_attention_to_sdpa(module: nn.Module, module_name: str = "") -> None:
    """
    Convert standard attention modules to use SDPA.
    
    This is a helper function that can be used to automatically convert
    existing attention layers to use the more memory-efficient SDPA.
    """
    for name, child in module.named_children():
        full_name = f"{module_name}.{name}" if module_name else name
        
        # Check if this is a MultiheadAttention module
        if isinstance(child, nn.MultiheadAttention):
            # Replace with SDPA version
            sdpa_attn = SDPAAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout if hasattr(child, 'dropout') else 0.0,
                bias=child.in_proj_bias is not None
            )
            
            # Copy weights if possible
            if hasattr(child, 'in_proj_weight') and child.in_proj_weight is not None:
                embed_dim = child.embed_dim
                # Split the combined QKV weight
                q_weight = child.in_proj_weight[:embed_dim]
                k_weight = child.in_proj_weight[embed_dim:2*embed_dim]
                v_weight = child.in_proj_weight[2*embed_dim:]
                
                sdpa_attn.q_proj.weight.data = q_weight
                sdpa_attn.k_proj.weight.data = k_weight
                sdpa_attn.v_proj.weight.data = v_weight
                
                if child.in_proj_bias is not None:
                    q_bias = child.in_proj_bias[:embed_dim]
                    k_bias = child.in_proj_bias[embed_dim:2*embed_dim]
                    v_bias = child.in_proj_bias[2*embed_dim:]
                    
                    sdpa_attn.q_proj.bias.data = q_bias
                    sdpa_attn.k_proj.bias.data = k_bias
                    sdpa_attn.v_proj.bias.data = v_bias
            
            if hasattr(child, 'out_proj'):
                sdpa_attn.out_proj.weight.data = child.out_proj.weight.data.clone()
                if child.out_proj.bias is not None:
                    sdpa_attn.out_proj.bias.data = child.out_proj.bias.data.clone()
            
            # Replace the module
            setattr(module, name, sdpa_attn)
            print(f"Converted {full_name} to SDPA attention")
        
        # Recursively process children
        convert_attention_to_sdpa(child, full_name)


# Example usage functions for common attention patterns
def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """
    Convenience function for direct SDPA usage.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Attention mask
        is_causal: Whether to use causal masking
        dropout_p: Dropout probability
        
    Returns:
        Attention output
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal
    )


def checkpointed_attention(
    attention_fn,
    *args,
    **kwargs
) -> torch.Tensor:
    """
    Apply attention with activation checkpointing.
    
    Args:
        attention_fn: Attention function to checkpoint
        *args: Arguments to attention function
        **kwargs: Keyword arguments to attention function
        
    Returns:
        Attention output
    """
    from torch.utils.checkpoint import checkpoint
    return checkpoint(attention_fn, *args, **kwargs)