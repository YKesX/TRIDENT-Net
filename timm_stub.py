"""
Simple timm stub for basic functionality.
"""
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_model(model_name, pretrained=True, features_only=False, **kwargs):
    """Stub for timm.create_model"""
    model = DummyModel(**kwargs)
    if features_only:
        return DummyFeatureModel()
    return model

class DummyFeatureModel(nn.Module):
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

# Make timm importable
import sys
sys.modules['timm'] = sys.modules[__name__]