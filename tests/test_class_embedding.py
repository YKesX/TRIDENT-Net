"""
Test class embedding and calibration dimension updates.

Tests Phase 2 of the hardening plan: 32-d class embedding and 1696-d CalibGLM input.

Author: Yağızhan Keskin
"""

import torch
import pytest
import sys
sys.path.append('.')

from trident.fusion_guard.cross_attn_fusion import CrossAttnFusion, ClassEmbedding
from trident.fusion_guard.calib_glm import CalibGLM


def test_class_embedding():
    """Test ClassEmbedding module."""
    num_classes = 100
    embed_dim = 32
    
    embedding = ClassEmbedding(num_classes, embed_dim)
    
    # Test normal class IDs
    class_ids = torch.tensor([0, 5, 10, 99])
    embeddings = embedding(class_ids)
    
    assert embeddings.shape == (4, 32), f"Expected shape (4, 32), got {embeddings.shape}"
    
    # Test out-of-vocabulary class IDs
    oov_class_ids = torch.tensor([100, 150, -1, 200])
    oov_embeddings = embedding(oov_class_ids)
    
    assert oov_embeddings.shape == (4, 32), f"Expected shape (4, 32), got {oov_embeddings.shape}"
    
    # All OOV embeddings should be the same (unknown_embedding)
    assert torch.allclose(oov_embeddings[0], oov_embeddings[1]), "OOV embeddings should be identical"
    assert torch.allclose(oov_embeddings[0], oov_embeddings[2]), "OOV embeddings should be identical"
    assert torch.allclose(oov_embeddings[0], oov_embeddings[3]), "OOV embeddings should be identical"
    
    # Mixed valid and invalid IDs
    mixed_ids = torch.tensor([5, 150, 10, -1])
    mixed_embeddings = embedding(mixed_ids)
    
    assert mixed_embeddings.shape == (4, 32), f"Expected shape (4, 32), got {mixed_embeddings.shape}"
    
    # Valid embeddings should be different from unknown embedding
    assert not torch.allclose(mixed_embeddings[0], mixed_embeddings[1]), "Valid and OOV embeddings should differ"
    assert not torch.allclose(mixed_embeddings[2], mixed_embeddings[3]), "Valid and OOV embeddings should differ"
    
    # OOV embeddings should be the same
    assert torch.allclose(mixed_embeddings[1], mixed_embeddings[3]), "OOV embeddings should be identical"
    
    print("✅ ClassEmbedding test passed")


def test_cross_attn_fusion_with_class_embedding():
    """Test CrossAttnFusion with class embedding support."""
    # Define dimensions according to tasks.yml
    dims = {'zi': 768, 'zt': 512, 'zr': 384, 'e_cls': 32}
    
    fusion = CrossAttnFusion(dims=dims, num_classes=100)
    
    batch_size = 2
    
    # Create test inputs
    zi = torch.randn(batch_size, dims['zi'])  # I-branch features
    zt = torch.randn(batch_size, dims['zt'])  # T-branch features  
    zr = torch.randn(batch_size, dims['zr'])  # R-branch features
    class_ids = torch.tensor([5, 10])  # Class IDs
    
    # Forward pass with class IDs
    z_fused, p_hit, p_kill, attn_maps, top_events = fusion(zi, zt, zr, class_ids=class_ids)
    
    # Check output shapes
    assert z_fused.shape == (batch_size, 512), f"Expected z_fused shape ({batch_size}, 512), got {z_fused.shape}"
    assert p_hit.shape == (batch_size, 1), f"Expected p_hit shape ({batch_size}, 1), got {p_hit.shape}"
    assert p_kill.shape == (batch_size, 1), f"Expected p_kill shape ({batch_size}, 1), got {p_kill.shape}"
    
    # Check probability ranges
    assert torch.all((p_hit >= 0) & (p_hit <= 1)), "p_hit should be in [0,1]"
    assert torch.all((p_kill >= 0) & (p_kill <= 1)), "p_kill should be in [0,1]"
    
    # Test without class IDs
    z_fused_no_cls, p_hit_no_cls, p_kill_no_cls, _, _ = fusion(zi, zt, zr)
    
    assert z_fused_no_cls.shape == (batch_size, 512), "Should work without class IDs"
    assert p_hit_no_cls.shape == (batch_size, 1), "Should work without class IDs"
    assert p_kill_no_cls.shape == (batch_size, 1), "Should work without class IDs"
    
    print("✅ CrossAttnFusion with class embedding test passed")


def test_calibration_features():
    """Test calibration feature extraction with correct dimensions."""
    dims = {'zi': 768, 'zt': 512, 'zr': 384, 'e_cls': 32}
    fusion = CrossAttnFusion(dims=dims, num_classes=100)
    
    batch_size = 3
    
    # Create test inputs
    zi = torch.randn(batch_size, dims['zi'])
    zt = torch.randn(batch_size, dims['zt'])
    zr = torch.randn(batch_size, dims['zr'])
    class_ids = torch.tensor([1, 50, 99])
    
    # Get calibration features
    calib_features = fusion.get_calibration_features(zi, zt, zr, class_ids=class_ids)
    
    # Check dimensions: 768 + 512 + 384 + 32 = 1696
    expected_dim = dims['zi'] + dims['zt'] + dims['zr'] + dims['e_cls']
    assert calib_features.shape == (batch_size, expected_dim), \
        f"Expected calibration features shape ({batch_size}, {expected_dim}), got {calib_features.shape}"
    
    assert expected_dim == 1696, f"Expected total dimension 1696, got {expected_dim}"
    
    # Test without class IDs (should use zero embeddings)
    calib_features_no_cls = fusion.get_calibration_features(zi, zt, zr)
    assert calib_features_no_cls.shape == (batch_size, expected_dim), \
        "Should work without class IDs using zero embeddings"
    
    # The class embedding part should be zeros when no class_ids provided
    class_part = calib_features_no_cls[:, -32:]  # Last 32 dimensions
    assert torch.allclose(class_part, torch.zeros_like(class_part)), \
        "Class embedding part should be zeros when no class_ids provided"
    
    print("✅ Calibration features test passed")


def test_calib_glm_1696_input():
    """Test CalibGLM accepts 1696-dimensional input."""
    calib_glm = CalibGLM(in_dim=1696, model="logreg", c=1.0, max_iter=200)
    
    # Check input dimension is stored correctly
    assert calib_glm.in_dim == 1696, f"Expected in_dim=1696, got {calib_glm.in_dim}"
    
    batch_size = 4
    features = torch.randn(batch_size, 1696)
    
    # Test forward pass (before fitting)
    try:
        p_hit_aux, p_kill_aux = calib_glm.forward(features)
        
        # Should return valid shapes
        assert p_hit_aux.shape == (batch_size, 1), f"Expected p_hit_aux shape ({batch_size}, 1), got {p_hit_aux.shape}"
        assert p_kill_aux.shape == (batch_size, 1), f"Expected p_kill_aux shape ({batch_size}, 1), got {p_kill_aux.shape}"
        
        # Probabilities should be in valid range
        assert torch.all((p_hit_aux >= 0) & (p_hit_aux <= 1)), "p_hit_aux should be in [0,1]"
        assert torch.all((p_kill_aux >= 0) & (p_kill_aux <= 1)), "p_kill_aux should be in [0,1]"
        
    except Exception as e:
        # If models aren't fitted, that's fine for this test
        if "not fitted" not in str(e).lower():
            raise e
    
    print("✅ CalibGLM 1696-d input test passed")


def test_full_pipeline_dimensions():
    """Test full pipeline with correct dimensions from CrossAttnFusion to CalibGLM."""
    dims = {'zi': 768, 'zt': 512, 'zr': 384, 'e_cls': 32}
    
    # Initialize modules
    fusion = CrossAttnFusion(dims=dims, num_classes=100)
    calib_glm = CalibGLM(in_dim=1696)
    
    batch_size = 2
    
    # Create test inputs
    zi = torch.randn(batch_size, dims['zi'])
    zt = torch.randn(batch_size, dims['zt'])
    zr = torch.randn(batch_size, dims['zr'])
    class_ids = torch.tensor([10, 25])
    
    # Get calibration features from fusion
    calib_features = fusion.get_calibration_features(zi, zt, zr, class_ids=class_ids)
    
    # Verify dimensions
    assert calib_features.shape == (batch_size, 1696), \
        f"Expected calibration features shape ({batch_size}, 1696), got {calib_features.shape}"
    
    # Test that CalibGLM can accept these features
    try:
        p_hit_aux, p_kill_aux = calib_glm.forward(calib_features)
        
        assert p_hit_aux.shape == (batch_size, 1), "CalibGLM should output correct p_hit_aux shape"
        assert p_kill_aux.shape == (batch_size, 1), "CalibGLM should output correct p_kill_aux shape"
        
    except Exception as e:
        # If models aren't fitted, that's expected
        if "not fitted" not in str(e).lower():
            raise e
    
    print("✅ Full pipeline dimensions test passed")


if __name__ == "__main__":
    test_class_embedding()
    test_cross_attn_fusion_with_class_embedding()
    test_calibration_features()
    test_calib_glm_1696_input()
    test_full_pipeline_dimensions()
    print("✅ All class embedding and calibration tests passed!")