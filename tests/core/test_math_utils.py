import pytest

math_utils = pytest.importorskip("synthstats.core.math_utils")
torch = pytest.importorskip("torch")
graph_connected_zero = math_utils.graph_connected_zero
sanitize_finite = math_utils.sanitize_finite


def test_finite_tensor_unchanged():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = sanitize_finite(t, fallback=0.0)
    assert torch.equal(result, t)


def test_nan_replaced():
    t = torch.tensor([float("nan")])
    result = sanitize_finite(t, fallback=-1.0)
    assert result.item() == -1.0


def test_inf_replaced():
    t = torch.tensor([float("inf"), float("-inf")])
    result = sanitize_finite(t, fallback=0.0)
    assert torch.equal(result, torch.tensor([0.0, 0.0]))


def test_mixed_finite_and_nan():
    t = torch.tensor([1.0, float("nan"), 3.0, float("inf")])
    result = sanitize_finite(t, fallback=-99.0)
    expected = torch.tensor([1.0, -99.0, 3.0, -99.0])
    assert torch.equal(result, expected)


def test_graph_connected_zero_without_inputs_returns_grad_enabled_scalar():
    result = graph_connected_zero()
    assert result.shape == torch.Size([])
    assert result.item() == 0.0
    assert result.requires_grad is True
