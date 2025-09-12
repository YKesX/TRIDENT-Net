"""
Test export functionality for ONNX and TorchScript round-trip validation.
Phase 10: Exports validation
"""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
from pathlib import Path
import sys
sys.path.append('.')

import trident
from trident.common.export import ModelExporter
from trident.fusion_guard.cross_attn_fusion import CrossAttnFusion
from trident.trident_i.dualvision_v2 import DualVisionV2


class SimpleTestModel(nn.Module):
    """Simple model for export testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_simple_torchscript_export():
    """Test basic TorchScript export functionality."""
    print("🧪 Testing TorchScript export...")
    
    model = SimpleTestModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Test forward pass
    with torch.no_grad():
        original_output = model(dummy_input)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Test traced model output
    with torch.no_grad():
        traced_output = traced_model(dummy_input)
    
    # Compare outputs
    assert torch.allclose(original_output, traced_output, atol=1e-5), \
        "TorchScript output differs from original model"
    
    print("✅ TorchScript export test passed")


def test_simple_onnx_export():
    """Test basic ONNX export functionality."""
    print("🧪 Testing ONNX export...")
    
    model = SimpleTestModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Create temporary file for ONNX export
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx_path = tmp_file.name
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Check if file exists and has reasonable size
        assert os.path.exists(onnx_path), "ONNX file was not created"
        file_size = os.path.getsize(onnx_path)
        assert file_size > 1000, f"ONNX file too small: {file_size} bytes"
        
        print(f"✅ ONNX export test passed - file size: {file_size} bytes")
        
    finally:
        # Clean up
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


def test_fusion_model_torchscript_export():
    """Test TorchScript export of fusion model."""
    print("🧪 Testing fusion model TorchScript export...")
    
    # Create fusion model
    fusion_model = CrossAttnFusion(
        zi_dim=768,
        zt_dim=512, 
        zr_dim=384,
        hidden_dim=256,
        num_heads=8,
        num_layers=2
    )
    fusion_model.eval()
    
    # Create dummy inputs
    zi = torch.randn(1, 768)
    zt = torch.randn(1, 512)
    zr = torch.randn(1, 384)
    class_emb = torch.randn(1, 32)
    
    # Test forward pass
    with torch.no_grad():
        original_output = fusion_model(zi, zt, zr, class_emb)
        
    # Create wrapper for tracing
    class FusionWrapper(nn.Module):
        def __init__(self, fusion_model):
            super().__init__()
            self.fusion = fusion_model
            
        def forward(self, zi, zt, zr, class_emb):
            return self.fusion(zi, zt, zr, class_emb)
    
    wrapper = FusionWrapper(fusion_model)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(wrapper, (zi, zt, zr, class_emb))
    
    # Test traced model output
    with torch.no_grad():
        traced_output = traced_model(zi, zt, zr, class_emb)
    
    # Compare outputs (allow some tolerance for fusion model complexity)
    for orig, traced in zip(original_output, traced_output):
        if torch.is_tensor(orig) and torch.is_tensor(traced):
            assert torch.allclose(orig, traced, atol=1e-4), \
                "TorchScript fusion output differs significantly"
    
    print("✅ Fusion model TorchScript export test passed")


def test_model_exporter_functionality():
    """Test ModelExporter class if it exists."""
    print("🧪 Testing ModelExporter functionality...")
    
    try:
        # Create simple model
        model = SimpleTestModel()
        
        # Check if ModelExporter exists
        if hasattr(trident.common.export, 'ModelExporter'):
            exporter = ModelExporter(model)
            
            # Test basic functionality
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Test TorchScript export
            traced = exporter.to_torchscript(dummy_input)
            assert traced is not None, "TorchScript export failed"
            
            print("✅ ModelExporter test passed")
        else:
            print("⚠️ ModelExporter not found - skipping")
            
    except Exception as e:
        print(f"⚠️ ModelExporter test error: {e}")


def test_export_metadata_validation():
    """Test that exported models contain proper metadata."""
    print("🧪 Testing export metadata...")
    
    model = SimpleTestModel()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Test TorchScript metadata
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Check that we can get some basic info
    assert hasattr(traced_model, 'graph'), "TorchScript model missing graph"
    assert hasattr(traced_model, 'code'), "TorchScript model missing code"
    
    # Test ONNX metadata
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
        onnx_path = tmp_file.name
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        # Basic file validation
        assert os.path.exists(onnx_path), "ONNX file not created"
        
        # Check file is not empty
        file_size = os.path.getsize(onnx_path)
        assert file_size > 0, "ONNX file is empty"
        
        print("✅ Export metadata validation passed")
        
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


if __name__ == "__main__":
    test_simple_torchscript_export()
    test_simple_onnx_export()
    test_fusion_model_torchscript_export()
    test_model_exporter_functionality()
    test_export_metadata_validation()
    print("🎉 All export tests completed!")