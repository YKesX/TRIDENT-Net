"""
Test hierarchy regularizer and fusion multitask loss.

Tests Phase 4 of the hardening plan: hierarchy constraint and fusion loss components.

Author: Yağızhan Keskin
"""

import torch
import pytest
import sys
sys.path.append('.')

from trident.common.losses import HierarchyRegularizer, BrierScore, FusionMultitaskLoss
from trident.common.metrics import auroc, f1, brier_score, expected_calibration_error
from trident.fusion_guard.cross_attn_fusion import CrossAttnFusion


def test_hierarchy_regularizer():
    """Test hierarchy regularizer enforces p_kill <= p_hit."""
    regularizer = HierarchyRegularizer(weight=0.2)
    
    # Test case 1: No violations (p_kill <= p_hit)
    p_hit = torch.tensor([[0.8], [0.6], [0.9]])
    p_kill = torch.tensor([[0.5], [0.4], [0.7]])  # All <= p_hit
    
    loss_no_violation = regularizer(p_hit, p_kill)
    assert loss_no_violation.item() == 0.0, f"Expected 0 loss with no violations, got {loss_no_violation.item()}"
    
    # Test case 2: With violations (p_kill > p_hit)
    p_hit = torch.tensor([[0.5], [0.6], [0.3]])
    p_kill = torch.tensor([[0.8], [0.4], [0.9]])  # [0.8>0.5, 0.4<=0.6, 0.9>0.3]
    
    loss_with_violations = regularizer(p_hit, p_kill)
    assert loss_with_violations.item() > 0.0, f"Expected positive loss with violations, got {loss_with_violations.item()}"
    
    # Manual calculation: violations = relu([0.3, 0, 0.6]) = [0.3, 0, 0.6]
    # Mean violation = (0.3 + 0 + 0.6) / 3 = 0.3
    # Loss = 0.2 * 0.3 = 0.06
    expected_loss = 0.2 * (0.3 + 0.0 + 0.6) / 3
    assert abs(loss_with_violations.item() - expected_loss) < 1e-6, \
        f"Expected loss {expected_loss}, got {loss_with_violations.item()}"
    
    print("✅ Hierarchy regularizer test passed")


def test_brier_score():
    """Test Brier score computation."""
    brier = BrierScore()
    
    # Perfect predictions
    p_pred = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    y_true = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    score_perfect = brier(p_pred, y_true)
    assert score_perfect.item() == 0.0, f"Expected 0 Brier score for perfect predictions, got {score_perfect.item()}"
    
    # Worst predictions
    p_pred = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    y_true = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    score_worst = brier(p_pred, y_true)
    assert score_worst.item() == 1.0, f"Expected 1.0 Brier score for worst predictions, got {score_worst.item()}"
    
    # Random predictions
    p_pred = torch.tensor([[0.7], [0.3], [0.8], [0.2]])
    y_true = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    score_random = brier(p_pred, y_true)
    # Manual: (0.7-1)^2 + (0.3-0)^2 + (0.8-1)^2 + (0.2-0)^2 = 0.09 + 0.09 + 0.04 + 0.04 = 0.26/4 = 0.065
    expected_random = ((0.7-1)**2 + (0.3-0)**2 + (0.8-1)**2 + (0.2-0)**2) / 4
    assert abs(score_random.item() - expected_random) < 1e-6, \
        f"Expected Brier score {expected_random}, got {score_random.item()}"
    
    print("✅ Brier score test passed")


def test_fusion_multitask_loss():
    """Test fusion multitask loss components."""
    loss_fn = FusionMultitaskLoss(
        bce_hit_weight=1.0,
        bce_kill_weight=1.0,
        brier_weight=0.25,
        hierarchy_weight=0.2
    )
    
    # Create test data with violations
    p_hit = torch.tensor([[0.6], [0.8]])
    p_kill = torch.tensor([[0.9], [0.5]])  # First sample violates p_kill > p_hit
    y_hit = torch.tensor([[1.0], [1.0]])
    y_kill = torch.tensor([[0.0], [1.0]])
    
    loss_dict = loss_fn(p_hit, p_kill, y_hit, y_kill)
    
    # Check all components are present
    expected_keys = ['total_loss', 'bce_hit', 'bce_kill', 'brier_hit', 'brier_kill', 'hierarchy_reg']
    for key in expected_keys:
        assert key in loss_dict, f"Missing loss component: {key}"
        assert isinstance(loss_dict[key], torch.Tensor), f"Loss component {key} should be tensor"
    
    # Check hierarchy regularizer is positive (since we have violations)
    assert loss_dict['hierarchy_reg'].item() > 0, "Hierarchy regularizer should be positive with violations"
    
    # Check total loss is combination of components
    expected_total = (
        1.0 * loss_dict['bce_hit'] +
        1.0 * loss_dict['bce_kill'] +
        0.25 * loss_dict['brier_hit'] +
        0.25 * loss_dict['brier_kill'] +
        loss_dict['hierarchy_reg']
    )
    
    assert torch.allclose(loss_dict['total_loss'], expected_total, atol=1e-6), \
        "Total loss should be sum of weighted components"
    
    print("✅ Fusion multitask loss test passed")


def test_fusion_loss_integration():
    """Test fusion loss integration with CrossAttnFusion."""
    dims = {'zi': 768, 'zt': 512, 'zr': 384, 'e_cls': 32}
    fusion = CrossAttnFusion(dims=dims, num_classes=100, d_model=64, n_layers=1, n_heads=2)
    
    batch_size = 3
    
    # Create test inputs
    zi = torch.randn(batch_size, dims['zi'])
    zt = torch.randn(batch_size, dims['zt'])
    zr = torch.randn(batch_size, dims['zr'])
    class_ids = torch.tensor([1, 5, 10])
    
    # Forward pass
    z_fused, p_hit, p_kill, attn_maps, top_events = fusion(zi, zt, zr, class_ids=class_ids)
    
    # Create dummy targets
    y_hit = torch.randint(0, 2, (batch_size, 1)).float()
    y_kill = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Test loss computation
    loss_dict = fusion.compute_loss(p_hit, p_kill, y_hit, y_kill)
    
    # Check loss components
    assert 'total_loss' in loss_dict, "Should have total_loss"
    assert loss_dict['total_loss'].item() >= 0, "Total loss should be non-negative"
    
    # Test metrics computation
    metrics = fusion.compute_metrics(p_hit, p_kill, y_hit, y_kill)
    
    # Check metrics are present
    expected_metrics = ['AUROC_hit', 'AUROC_kill', 'F1_hit', 'F1_kill', 
                       'Brier_hit', 'Brier_kill', 'ECE_hit', 'ECE_kill',
                       'hierarchy_violation_rate', 'hierarchy_violation_magnitude']
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    # Check hierarchy violation metrics
    assert 0 <= metrics['hierarchy_violation_rate'] <= 1, "Violation rate should be in [0,1]"
    assert metrics['hierarchy_violation_magnitude'] >= 0, "Violation magnitude should be non-negative"
    
    print("✅ Fusion loss integration test passed")


def test_hierarchy_constraint_enforcement():
    """Test that hierarchy constraint actually works in practice."""
    # Create a simple scenario where we can enforce the constraint
    
    # Start with violations
    p_hit = torch.tensor([[0.3], [0.5], [0.2]], requires_grad=True)
    p_kill = torch.tensor([[0.8], [0.6], [0.9]], requires_grad=True)  # All violate p_kill > p_hit
    
    # Create regularizer with high weight
    regularizer = HierarchyRegularizer(weight=10.0)
    
    # Optimizer to minimize hierarchy violations
    optimizer = torch.optim.SGD([p_hit, p_kill], lr=0.1)
    
    initial_loss = regularizer(p_hit, p_kill)
    
    # Several optimization steps
    for _ in range(20):
        optimizer.zero_grad()
        loss = regularizer(p_hit, p_kill)
        loss.backward()
        optimizer.step()
        
        # Clamp probabilities to [0, 1]
        with torch.no_grad():
            p_hit.clamp_(0, 1)
            p_kill.clamp_(0, 1)
    
    final_loss = regularizer(p_hit, p_kill)
    
    # Loss should decrease (violations should reduce)
    assert final_loss.item() < initial_loss.item(), \
        f"Hierarchy loss should decrease from {initial_loss.item()} to {final_loss.item()}"
    
    # Check that violations are reduced
    with torch.no_grad():
        violations = torch.relu(p_kill - p_hit)
        violation_count = (violations > 0.1).sum().item()  # Small tolerance
    
    assert violation_count <= 1, f"Should have at most 1 significant violation, got {violation_count}"
    
    print("✅ Hierarchy constraint enforcement test passed")


def test_metrics_functions():
    """Test metric computation functions."""
    # Create test data
    y_true = torch.tensor([[1], [0], [1], [0], [1]]).float()
    y_prob = torch.tensor([[0.9], [0.1], [0.8], [0.2], [0.7]]).float()
    
    # Test F1
    f1_score = f1(y_true, y_prob, threshold=0.5)
    assert 0 <= f1_score <= 1, f"F1 score should be in [0,1], got {f1_score}"
    
    # Test Brier
    brier = brier_score(y_true, y_prob)
    assert 0 <= brier <= 1, f"Brier score should be in [0,1], got {brier}"
    
    # Test ECE
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece}"
    
    print("✅ Metrics functions test passed")


if __name__ == "__main__":
    test_hierarchy_regularizer()
    test_brier_score()
    test_fusion_multitask_loss()
    test_fusion_loss_integration()
    test_hierarchy_constraint_enforcement()
    test_metrics_functions()
    print("✅ All hierarchy and fusion loss tests passed!")