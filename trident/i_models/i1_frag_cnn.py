"""
TRIDENT-I1: Fragment Detection CNN (U-Net-like segmentation)

Author: Yağızhan Keskin
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    # Simple fallback for timm
    class timm:
        @staticmethod
        def create_model(model_name, pretrained=True, features_only=False, **kwargs):
            # Simple dummy model
            if features_only:
                class FeatureModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
                        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
                    def forward(self, x):
                        f1 = self.conv1(x)
                        f2 = self.conv2(f1)
                        f3 = self.conv3(f2)
                        return [f1, f2, f3]
                return FeatureModel()
            else:
                return nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 1000)
                )

from ..common.types import BranchModule, EventToken, FeatureVec


class UNetDecoder(nn.Module):
    """U-Net decoder for segmentation."""
    
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        super().__init__()
        
        # Reverse encoder channels for decoder
        encoder_channels = encoder_channels[::-1]
        
        # Create decoder blocks
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[i] if i == 0 else decoder_channels[i-1]
            skip_ch = encoder_channels[i+1] if i+1 < len(encoder_channels) else 0
            out_ch = decoder_channels[i]
            
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Reverse features (deepest first)
        features = features[::-1]
        
        x = features[0]
        for i, block in enumerate(self.blocks):
            skip = features[i+1] if i+1 < len(features) else None
            x = block(x, skip)
        
        return x


class DecoderBlock(nn.Module):
    """Single decoder block with skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        
        conv_in_ch = in_channels // 2 + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_ch, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        
        if skip is not None:
            # Match spatial dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class FragCNN(BranchModule):
    """
    Fragment Detection CNN using U-Net architecture.
    
    Detects and segments debris/fragments in visible imagery.
    Uses EfficientNet or MobileNet as encoder backbone.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        channels: int = 3,
        out_dim: int = 256,
        num_classes: int = 1,
    ):
        super().__init__(out_dim)
        
        self.backbone_name = backbone
        self.channels = channels
        self.num_classes = num_classes
        
        # Create encoder backbone
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            in_chans=channels,
        )
        
        # Get encoder channel dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, 224, 224)
            encoder_features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in encoder_features]
        
        # Create decoder
        decoder_channels = [256, 128, 64, 32, 16]
        self.decoder = UNetDecoder(encoder_channels, decoder_channels)
        
        # Segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        
        # Global feature extraction for FeatureVec
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(encoder_channels[-1], out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim),
        )
        
        # Event detection heads
        self.debris_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.spread_head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
    
    def forward(self, rgb_roi: torch.Tensor) -> Tuple[FeatureVec, List[EventToken]]:
        """
        Forward pass for fragment detection.
        
        Args:
            rgb_roi: RGB image tensor of shape (B, C, H, W)
            
        Returns:
            tuple: (FeatureVec with shape (B, out_dim), list of EventTokens)
        """
        batch_size = rgb_roi.shape[0]
        
        # Encoder forward pass
        encoder_features = self.encoder(rgb_roi)
        
        # Decoder for segmentation
        decoder_output = self.decoder(encoder_features)
        segmentation = self.seg_head(decoder_output)
        
        # Global feature extraction
        global_features = self.global_pool(encoder_features[-1]).flatten(1)
        features = self.feature_head(global_features)
        
        # Event detection
        debris_prob = self.debris_head(features)
        spread_magnitude = self.spread_head(features)
        
        # Create FeatureVec
        feature_vec = FeatureVec(z=features)
        
        # Create EventTokens
        events = []
        for b in range(batch_size):
            # Debris detection event
            if debris_prob[b, 0] > 0.5:
                events.append(EventToken(
                    type="debris_detected",
                    value=debris_prob[b, 0].item(),
                    t_start=0.0,
                    t_end=1.0,
                    quality=debris_prob[b, 0].item(),
                    meta={
                        "segmentation_area": torch.sigmoid(segmentation[b]).sum().item(),
                        "max_intensity": torch.sigmoid(segmentation[b]).max().item(),
                    }
                ))
            
            # Spread analysis event
            spread_val = spread_magnitude[b, 0].item()
            if abs(spread_val) > 0.3:
                events.append(EventToken(
                    type="spread_pattern",
                    value=spread_val,
                    t_start=0.0,
                    t_end=1.0,
                    quality=min(abs(spread_val), 1.0),
                    meta={
                        "direction": "expansion" if spread_val > 0 else "contraction",
                        "magnitude": abs(spread_val),
                    }
                ))
        
        # Store intermediate outputs for potential use
        self._last_segmentation = segmentation
        self._last_features = encoder_features
        
        return feature_vec, events
    
    def get_segmentation_output(self) -> torch.Tensor:
        """Get the last segmentation output (for loss computation)."""
        if hasattr(self, '_last_segmentation'):
            return self._last_segmentation
        else:
            raise RuntimeError("No segmentation output available. Run forward() first.")
    
    def get_feature_maps(self) -> List[torch.Tensor]:
        """Get encoder feature maps for visualization."""
        if hasattr(self, '_last_features'):
            return self._last_features
        else:
            raise RuntimeError("No features available. Run forward() first.")


def create_frag_cnn(config: dict) -> FragCNN:
    """Factory function to create FragCNN from config."""
    return FragCNN(
        backbone=config.get("backbone", "efficientnet_b0"),
        channels=config.get("channels", 3),
        out_dim=config.get("out_dim", 256),
        num_classes=config.get("num_classes", 1),
    )