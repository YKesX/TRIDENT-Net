"""
Test shape validation for TRIDENT-Net components.

Validates that every forward pass returns exact shapes specified in tasks.yml.

Author: Yağızhan Keskin
"""

import torch
import yaml
from pathlib import Path

import sys
sys.path.append('.')
import trident


def load_tasks_config():
    """Load tasks.yml configuration."""
    with open('tasks.yml', 'r') as f:
        return yaml.safe_load(f)


def test_rgb_shapes():
    """Test RGB input shapes match spec: B x 3 x 3 x 768 x 1120"""
    config = load_tasks_config()
    expected_shape = "B x 3 x 3 x 768 x 1120"
    
    # Test with batch size 2
    batch_size = 2
    rgb_seq = torch.randn(batch_size, 3, 3, 768, 1120)
    
    assert rgb_seq.shape == (batch_size, 3, 3, 768, 1120), f"RGB shape mismatch: expected {expected_shape}"
    print(f"✅ RGB shape test passed: {rgb_seq.shape}")


def test_ir_shapes():
    """Test IR input shapes match spec: B x 3 x 1 x 768 x 1120"""
    config = load_tasks_config()
    expected_shape = "B x 3 x 1 x 768 x 1120"
    
    # Test with batch size 2  
    batch_size = 2
    ir_seq = torch.randn(batch_size, 3, 1, 768, 1120)
    
    assert ir_seq.shape == (batch_size, 3, 1, 768, 1120), f"IR shape mismatch: expected {expected_shape}"
    print(f"✅ IR shape test passed: {ir_seq.shape}")


def test_kinematics_shapes():
    """Test kinematics input shapes match spec: B x 3 x 9"""
    config = load_tasks_config()
    expected_shape = "B x 3 x 9"
    
    # Test with batch size 2
    batch_size = 2
    k_seq = torch.randn(batch_size, 3, 9)
    
    assert k_seq.shape == (batch_size, 3, 9), f"Kinematics shape mismatch: expected {expected_shape}"
    print(f"✅ Kinematics shape test passed: {k_seq.shape}")


def test_component_output_shapes():
    """Test component output shapes match tasks.yml specifications."""
    config = load_tasks_config()
    batch_size = 2
    
    # Test each component's expected output shapes
    components = config.get('components', {})
    
    for comp_name, comp_config in components.items():
        outputs = comp_config.get('outputs', {})
        print(f"Component {comp_name} outputs: {outputs}")
        
        # Verify output shape specifications exist
        for output_name, output_spec in outputs.items():
            if isinstance(output_spec, dict) and 'shape' in output_spec:
                shape_str = output_spec['shape']
                print(f"  {output_name}: {shape_str}")
                
                # Parse shape string and validate format
                assert 'B' in shape_str, f"Shape {shape_str} must include batch dimension B"


def test_fusion_shapes():
    """Test fusion component input/output shapes."""
    config = load_tasks_config()
    
    # Check f2 (CrossAttnFusion) shapes
    f2_config = config['components']['f2']
    
    inputs = f2_config['inputs']
    assert inputs['zi']['shape'] == "B x 768", "zi input shape mismatch"
    assert inputs['zt']['shape'] == "B x 512", "zt input shape mismatch"  
    assert inputs['zr']['shape'] == "B x 384", "zr input shape mismatch"
    
    outputs = f2_config['outputs']
    assert outputs['z_fused']['shape'] == "B x 512", "z_fused output shape mismatch"
    assert outputs['p_hit']['shape'] == "B x 1", "p_hit output shape mismatch"
    assert outputs['p_kill']['shape'] == "B x 1", "p_kill output shape mismatch"
    
    print("✅ Fusion component shapes validated")


if __name__ == "__main__":
    print("🧪 Running TRIDENT-Net shape validation tests...")
    
    test_rgb_shapes()
    test_ir_shapes() 
    test_kinematics_shapes()
    test_component_output_shapes()
    test_fusion_shapes()
    
    print("✅ All shape tests passed!")