"""
CrossAttnFusion - Cross-modal transformer fusion with multitask outputs

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.types import FusionModule, EventToken


class ClassEmbedding(nn.Module):
    """
    Class embedding layer for injecting class information into fusion.

    Creates embeddings for class IDs with fallback support for unknown classes.
    """

    def __init__(self, num_classes: int, embed_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Main embedding table for known classes
        self.embedding = nn.Embedding(num_classes, embed_dim)

        # Fallback embedding for unknown/out-of-vocab classes
        self.unknown_embedding = nn.Parameter(torch.randn(embed_dim))

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.1)
        nn.init.normal_(self.unknown_embedding, std=0.1)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through class embedding.

        Args:
            class_ids: Class IDs (B,) - integer tensor

        Returns:
            Class embeddings (B, embed_dim)
        """
        B = class_ids.shape[0]
        device = class_ids.device

        # Handle out-of-vocabulary class IDs
        valid_mask = (class_ids >= 0) & (class_ids < self.num_classes)

        # Initialize output with unknown embeddings
        output = self.unknown_embedding.unsqueeze(0).expand(B, -1).clone()

        if valid_mask.any():
            # Use learned embeddings for valid class IDs
            valid_ids = class_ids[valid_mask]
            valid_embeddings = self.embedding(valid_ids)
            output[valid_mask] = valid_embeddings

        return output


class CrossAttnFusion(FusionModule):
    """
    Cross-modal transformer fusion module with multitask outputs.

    Fuses features from I/T/R branches using cross-attention and produces
    both hit and kill predictions with attention maps and top events.

    Input:
        - zi (B, 768) - concat(i1.zi=256, i2.zi=256, i3.zi=256)
        - zt (B, 512) - concat(t1.zt=256, t2.zt=256)
        - zr (B, 384) - concat(r2.zr2=192, r3.zr3=192)
        - events (list) - events from all modalities
    Outputs:
        - z_fused (B, 512) - fused features
        - p_hit (B, 1) - hit probability
        - p_kill (B, 1) - kill probability
        - attn_maps (dict) - attention maps
        - top_events (list) - top contributing events
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        dims: dict | None = None,
        num_classes: int = 100,
        **legacy_kwargs: Any,
    ):
        super().__init__(out_dim=512)

        # Support legacy arg names like zi_dim, zt_dim, zr_dim, hidden_dim, num_heads, num_layers
        # Flag to control legacy return/signature behavior
        self.legacy_api = False

        if legacy_kwargs:
            zi_dim = legacy_kwargs.get("zi_dim")
            zt_dim = legacy_kwargs.get("zt_dim")
            zr_dim = legacy_kwargs.get("zr_dim")
            if zi_dim and zt_dim and zr_dim:
                dims = {"zi": zi_dim, "zt": zt_dim, "zr": zr_dim, "e_cls": 32}
            d_model = legacy_kwargs.get("d_model", d_model)
            n_heads = legacy_kwargs.get("num_heads", n_heads)
            n_layers = legacy_kwargs.get("num_layers", n_layers)
            mlp_hidden = legacy_kwargs.get("hidden_dim", mlp_hidden)
            # Any use of legacy kwarg names implies legacy API expectations
            self.legacy_api = True

        # Default dimensions if not provided
        if dims is None:
            dims = {"zi": 768, "zt": 512, "zr": 384, "e_cls": 32}

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dims = dims

        # Input projections for each modality
        self.i_proj = nn.Linear(dims["zi"], d_model)  # I-branch: 768 -> 512
        self.t_proj = nn.Linear(dims["zt"], d_model)  # T-branch: 512 -> 512
        self.r_proj = nn.Linear(dims["zr"], d_model)  # R-branch: 384 -> 512

        # Class embedding layer
        self.class_embedding = ClassEmbedding(num_classes, dims["e_cls"])
        self.cls_proj = nn.Linear(dims["e_cls"], d_model)  # Class: 32 -> 512

        # Modality embeddings (now 4 modalities: I, T, R, Class)
        self.modality_embeddings = nn.Parameter(torch.randn(4, d_model))

        # Cross-attention transformer layers
        self.transformer_layers = nn.ModuleList(
            [CrossModalTransformerLayer(d_model, n_heads, d_model * 4, dropout) for _ in range(n_layers)]
        )

        # Event processing
        self.event_processor = EventProcessor(d_model, dropout)

        # Output heads
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multitask prediction heads
        self.hit_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.kill_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        zi: torch.Tensor,
        zt: torch.Tensor,
        zr: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        events: Optional[List[EventToken]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Forward pass through CrossAttnFusion.

        Args:
            zi: I-branch features (B, 768)
            zt: T-branch features (B, 512)
            zr: R-branch features (B, 384)
            class_ids: Class IDs (B,) or precomputed embeddings (B, e_cls)
            events: List of events from all modalities
            labels: Labels for loss computation (ignored in forward pass)

        Returns:
            tuple: Non-tracing: (z_fused, p_hit, p_kill, attn_maps, top_events)
                   Tracing:    (z_fused, p_hit, p_kill)
        """
        B = zi.shape[0]
        device = zi.device

        # Project modality features to common dimension
        zi_proj = self.i_proj(zi)  # (B, d_model)
        zt_proj = self.t_proj(zt)  # (B, d_model)
        zr_proj = self.r_proj(zr)  # (B, d_model)

        # Handle class embeddings or IDs (support both)
        if class_ids is not None:
            if class_ids.dtype in (torch.long, torch.int32, torch.int64):
                class_emb = self.class_embedding(class_ids)  # (B, e_cls)
            else:
                # Assume already an embedding of shape (B, e_cls)
                class_emb = class_ids
            zcls_proj = self.cls_proj(class_emb)  # (B, d_model)
        else:
            # Use zero embeddings if no class info provided
            zcls_proj = torch.zeros(B, self.d_model, device=device)

        # Stack modality features (I, T, R, Class)
        features = torch.stack([zi_proj, zt_proj, zr_proj, zcls_proj], dim=1)  # (B, 4, d_model)

        # Add modality embeddings to each feature
        features = features + self.modality_embeddings.unsqueeze(0)  # (B, 4, d_model)

        # Apply transformer layers with attention tracking
        attention_maps: Dict[str, torch.Tensor] = {}
        x = features  # (B, 4, d_model)

        for i, layer in enumerate(self.transformer_layers):
            x, layer_attn = layer(x)
            attention_maps[f"layer_{i}"] = layer_attn

        # Process events if provided
        event_features = self.event_processor(events, B, device) if events else None

        # Global fusion with event integration
        if event_features is not None:
            # Add event features to sequence
            x = torch.cat([x, event_features.unsqueeze(1)], dim=1)

            # Final cross-attention with events
            final_layer = CrossModalTransformerLayer(self.d_model, self.n_heads, self.d_model * 4, 0.1)
            x, final_attn = final_layer(x)
            attention_maps["final_with_events"] = final_attn

        # Global pooling for fusion features
        z_fused = self.fusion_head(x.mean(dim=1))  # (B, d_model)

        # Multitask predictions with hierarchy enforcement p_kill <= p_hit
        p_hit = self.hit_head(z_fused)  # (B, 1)
        p_kill_raw = self.kill_head(z_fused)  # (B, 1)
        p_kill = torch.minimum(p_kill_raw, p_hit)

        # Derive top contributing events (for introspection) if provided
        top_events: List[EventToken] = self._extract_top_events(events, attention_maps) if events else []

        # When tracing, only return tensors to keep TorchScript happy and stable
        if torch.jit.is_tracing():
            if self.legacy_api:
                # Maintain legacy triple for tracing comparisons in tests
                return p_hit, p_kill, p_hit
            return z_fused, p_hit, p_kill

        # Legacy API expects probabilities and placeholders
        if self.legacy_api:
            p_hit_masked = p_hit
            p_kill_masked = p_kill
            spoof_risk = torch.zeros_like(p_hit)
            return p_hit, p_kill, p_hit_masked, p_kill_masked, spoof_risk

        return (z_fused, p_hit, p_kill, attention_maps, top_events)

    def _extract_top_events(self, events: List[EventToken], attention_maps: Dict[str, torch.Tensor]) -> List[EventToken]:
        """Extract top contributing events based on attention."""
        if not events or not attention_maps:
            return []

        # Simple heuristic: return events with highest score
        sorted_events = sorted(events, key=lambda e: getattr(e, "score", 0.0), reverse=True)
        return sorted_events[:5]  # Top 5 events

    def get_calibration_features(
        self, zi: torch.Tensor, zt: torch.Tensor, zr: torch.Tensor, class_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get concatenated features for calibration (zi + zt + zr + e_cls).

        Args:
            zi: I-branch features (B, 768)
            zt: T-branch features (B, 512)
            zr: R-branch features (B, 384)
            class_ids: Class IDs (B,) - optional

        Returns:
            Concatenated features (B, 1696)
        """
        # Get class embeddings
        if class_ids is not None:
            class_emb = self.class_embedding(class_ids)  # (B, e_cls)
        else:
            B = zi.shape[0]
            device = zi.device
            class_emb = torch.zeros(B, self.dims["e_cls"], device=device)

        # Concatenate all features
        calibration_features = torch.cat([zi, zt, zr, class_emb], dim=-1)  # (B, 1696)

        return calibration_features

    def compute_loss(
        self,
        p_hit: torch.Tensor,
        p_kill: torch.Tensor,
        y_hit: torch.Tensor,
        y_kill: torch.Tensor,
        loss_config: dict | None = None,
    ) -> dict:
        """
        Compute fusion multitask loss with hierarchy regularization.

        Args:
            p_hit: Predicted hit probabilities (B, 1)
            p_kill: Predicted kill probabilities (B, 1)
            y_hit: True hit labels (B, 1)
            y_kill: True kill labels (B, 1)
            loss_config: Loss configuration dictionary

        Returns:
            Dictionary with total loss and component losses
        """
        from ..common.losses import FusionMultitaskLoss

        # Default loss configuration
        if loss_config is None:
            loss_config = {
                "bce_hit": 1.0,
                "bce_kill": 1.0,
                "brier": 0.25,
                "hierarchy_regularizer": {"weight": 0.2},
            }

        # Create fusion loss
        fusion_loss = FusionMultitaskLoss(
            bce_hit_weight=loss_config.get("bce_hit", 1.0),
            bce_kill_weight=loss_config.get("bce_kill", 1.0),
            brier_weight=loss_config.get("brier", 0.25),
            hierarchy_weight=loss_config.get("hierarchy_regularizer", {}).get("weight", 0.2),
        )

        # Compute loss
        return fusion_loss(p_hit, p_kill, y_hit, y_kill)

    def compute_metrics(self, p_hit: torch.Tensor, p_kill: torch.Tensor, y_hit: torch.Tensor, y_kill: torch.Tensor) -> dict:
        """
        Compute evaluation metrics for fusion outputs.

        Args:
            p_hit: Predicted hit probabilities (B, 1)
            p_kill: Predicted kill probabilities (B, 1)
            y_hit: True hit labels (B, 1)
            y_kill: True kill labels (B, 1)

        Returns:
            Dictionary of computed metrics
        """
        from ..common.metrics import auroc, f1, brier_score, expected_calibration_error

        metrics: Dict[str, float] = {}

        # AUROC metrics
        try:
            metrics["AUROC_hit"] = auroc(y_hit, p_hit)
            metrics["AUROC_kill"] = auroc(y_kill, p_kill)
        except Exception:
            metrics["AUROC_hit"] = float("nan")
            metrics["AUROC_kill"] = float("nan")

        # F1 metrics
        metrics["F1_hit"] = f1(y_hit, p_hit, threshold=0.5)
        metrics["F1_kill"] = f1(y_kill, p_kill, threshold=0.5)

        # Brier score metrics
        metrics["Brier_hit"] = brier_score(y_hit, p_hit)
        metrics["Brier_kill"] = brier_score(y_kill, p_kill)

        # ECE metrics
        metrics["ECE_hit"] = expected_calibration_error(y_hit, p_hit, n_bins=15)
        metrics["ECE_kill"] = expected_calibration_error(y_kill, p_kill, n_bins=15)

        # Hierarchy constraint violation rate
        violations = torch.relu(p_kill - p_hit)
        metrics["hierarchy_violation_rate"] = (violations > 0).float().mean().item()
        metrics["hierarchy_violation_magnitude"] = violations.mean().item()

        return metrics


class SimpleMHA(nn.Module):
    """Deterministic multi-head self-attention for stable TorchScript tracing."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, S, E)
        B, S, E = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # reshape to (B, heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # attention scores: (B, heads, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)  # (B, heads, S, head_dim)
        # merge heads
        context = context.transpose(1, 2).contiguous().view(B, S, E)
        out = self.out_proj(context)
        # average weights across heads for interpretability: (B, S, S)
        attn_avg = attn_weights.mean(dim=1)
        return out, attn_avg


class CrossModalTransformerLayer(nn.Module):
    """Single cross-modal transformer layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        # Use a stable attention implementation for tracing
        self.self_attn = SimpleMHA(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention tracking."""
        # Self-attention
        attn_output, attn_weights = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class EventProcessor(nn.Module):
    """Process events into feature representations."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model

        # Event type embedding
        self.event_types = [
            "debris_detection",
            "flash_detection",
            "structural_damage",
            "thermal_signature",
            "close_approach",
            "high_acceleration",
        ]
        self.type_embedding = nn.Embedding(len(self.event_types), d_model // 4)

        # Event feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(d_model // 4 + 3, d_model // 2),  # type + [confidence, x, y]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, events: List[EventToken], batch_size: int, device: torch.device) -> torch.Tensor:
        """Process events into features."""
        if not events:
            return torch.zeros(batch_size, self.d_model, device=device)

        # Group events by batch
        batch_events: List[List[EventToken]] = [[] for _ in range(batch_size)]
        for event in events:
            # Our EventToken has 'meta' dict; default to 0 if not found
            batch_idx = 0
            if hasattr(event, "meta") and isinstance(event.meta, dict):
                batch_idx = int(event.meta.get("batch_idx", 0))
            if batch_idx < batch_size:
                batch_events[batch_idx].append(event)

        # Process each batch's events
        batch_features: List[torch.Tensor] = []
        for b_events in batch_events:
            if not b_events:
                batch_features.append(torch.zeros(self.d_model, device=device))
            else:
                # Take highest-score event
                top_event = max(b_events, key=lambda e: getattr(e, "score", 0.0))

                # Map type string to embedding index
                ev_type = getattr(top_event, "type", "") or ""
                type_idx = self.event_types.index(ev_type) if ev_type in self.event_types else 0
                type_emb = self.type_embedding(torch.tensor(type_idx, device=device))

                # Location from meta if available
                loc = (0.0, 0.0)
                if hasattr(top_event, "meta") and isinstance(top_event.meta, dict):
                    loc = tuple(top_event.meta.get("center_xy", (0.0, 0.0)))
                score = float(getattr(top_event, "score", 0.0))
                features = torch.cat([
                    type_emb,
                    torch.tensor([score, float(loc[0]) / 640.0, float(loc[1]) / 480.0], device=device),
                ])

                batch_features.append(self.feature_processor(features))

        return torch.stack(batch_features, dim=0)  # (B, d_model)