"""
FragCNN: U-Net-like encoder-decoder for segmentation and feature extraction.

Processes RGB ROI images to generate segmentation masks and pooled features.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ..common.types import BranchModule, EventToken, FeatureVec


class FragCNN(BranchModule):
    """
    Fragment detection CNN with U-Net architecture.
    
    Uses MobileNetV3 or EfficientNet encoder with skip connections
    for segmentation and pooled feature extraction.
    """
    
    def __init__(
        self,
        encoder_name: str = "mobilenetv3_large_100",
        out_dim: int = 256,
        num_classes: int = 2,
        pretrained: bool = True
    ) -> None:
        """
        Initialize FragCNN.
        
        Args:
            encoder_name: Timm model name for encoder
            out_dim: Output feature dimension
            num_classes: Number of segmentation classes
            pretrained: Use pretrained encoder weights
        """
        super().__init__()
        
        self.out_dim = out_dim
        self.num_classes = num_classes
        
        # Encoder (MobileNetV3 or EfficientNet)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # Multi-scale features
        )
        
        # Get encoder feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            encoder_features = self.encoder(dummy_input)
        self.encoder_dims = [f.shape[1] for f in encoder_features]
        
        # Decoder with skip connections
        self.decoder = UNetDecoder(
            encoder_dims=self.encoder_dims,
            decoder_dim=256,
            num_classes=num_classes
        )
        
        # Feature pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.encoder_dims[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
        
        # Event detection heads
        self.debris_detector = nn.Sequential(
            nn.Conv2d(num_classes, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.spread_estimator = nn.Sequential(
            nn.Conv2d(num_classes, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_roi: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Process RGB ROI and extract features + events.
        
        Args:
            rgb_roi: RGB image tensor (B, 3, H, W)
            
        Returns:
            Tuple of (feature_vector, list_of_events)
        """
        batch_size = rgb_roi.shape[0]
        
        # Encoder forward pass
        encoder_features = self.encoder(rgb_roi)
        
        # Decoder for segmentation
        segmentation_logits = self.decoder(encoder_features)
        segmentation_probs = torch.softmax(segmentation_logits, dim=1)
        
        # Global feature extraction
        pooled_features = self.global_pool(encoder_features[-1]).squeeze(-1).squeeze(-1)
        projected_features = self.feature_proj(pooled_features)
        
        # Event detection
        debris_score = self.debris_detector(segmentation_probs)
        spread_score = self.spread_estimator(segmentation_probs)
        
        # Create feature vector
        feature_vec = FeatureVec(z=projected_features)
        
        # Generate event tokens
        events = []
        for b in range(batch_size):
            # Debris detection event
            if debris_score[b] > 0.5:
                events.append(EventToken(
                    type="debris_detected",
                    value=debris_score[b].item(),
                    t_start=0.0,
                    t_end=1.0,
                    quality=debris_score[b].item(),
                    meta={"source": "frag_cnn", "batch_idx": b}
                ))
            
            # Expansion/spread event
            if spread_score[b] > 0.3:
                events.append(EventToken(
                    type="expansion",
                    value=spread_score[b].item(),
                    t_start=0.0,
                    t_end=1.0,
                    quality=spread_score[b].item(),
                    meta={"source": "frag_cnn", "batch_idx": b}
                ))
        
        return feature_vec, events
    
    def get_segmentation(self, rgb_roi: torch.Tensor) -> torch.Tensor:
        """
        Get segmentation mask for visualization.
        
        Args:
            rgb_roi: RGB image tensor (B, 3, H, W)
            
        Returns:
            Segmentation probabilities (B, C, H, W)
        """
        encoder_features = self.encoder(rgb_roi)
        segmentation_logits = self.decoder(encoder_features)
        return torch.softmax(segmentation_logits, dim=1)


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections."""
    
    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dim: int = 256,
        num_classes: int = 2
    ) -> None:
        """
        Initialize U-Net decoder.
        
        Args:
            encoder_dims: Encoder feature dimensions
            decoder_dim: Decoder channel dimension
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.decoder_dim = decoder_dim
        
        # Decoder blocks (bottom-up)
        self.decoder_blocks = nn.ModuleList()
        
        # Start from the deepest features
        current_dim = encoder_dims[-1]
        
        for i in range(len(encoder_dims) - 1, 0, -1):
            skip_dim = encoder_dims[i - 1]
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=current_dim,
                    skip_channels=skip_dim,
                    out_channels=decoder_dim
                )
            )
            current_dim = decoder_dim
        
        # Final classification head
        self.final_conv = nn.Conv2d(decoder_dim, num_classes, 1)
    
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode features to segmentation map.
        
        Args:
            encoder_features: List of encoder features (low to high resolution)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Start with deepest features
        x = encoder_features[-1]
        
        # Decode with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_features = encoder_features[-(i + 2)]  # Corresponding skip connection
            x = decoder_block(x, skip_features)
        
        # Final classification
        output = self.final_conv(x)
        
        # Upsample to input resolution
        target_size = encoder_features[0].shape[-2:]
        if output.shape[-2:] != target_size:
            output = F.interpolate(
                output, size=target_size, mode="bilinear", align_corners=False
            )
        
        return output


class DecoderBlock(nn.Module):
    """Single decoder block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ) -> None:
        """
        Initialize decoder block.
        
        Args:
            in_channels: Input channels from previous layer
            skip_channels: Skip connection channels
            out_channels: Output channels
        """
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
            
        Returns:
            Decoded features
        """
        # Upsample
        x = self.upsample(x)
        
        # Match spatial dimensions if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = self.conv(x)
        
        return x


# Default configuration
FRAG_CNN_CONFIG = {
    "encoder_name": "mobilenetv3_large_100",
    "out_dim": 256,
    "num_classes": 2,
    "pretrained": True
}