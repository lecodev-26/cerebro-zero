"""
Tests de estabilidad numérica
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor


def test_softmax_stability_large():
    x = Tensor([1000.0, 1001.0, 1002.0])
    sm = x.softmax()
    assert not np.any(np.isnan(sm.data))
    assert not np.any(np.isinf(sm.data))
    assert np.isclose(np.sum(sm.data), 1.0)


def test_softmax_stability_negative():
    x = Tensor([-1000.0, -1001.0, -1002.0])
    sm = x.softmax()
    assert not np.any(np.isnan(sm.data))
    assert np.isclose(np.sum(sm.data), 1.0)


def test_sigmoid_stability_large_positive():
    x = Tensor([500.0, 1000.0])
    s = x.sigmoid()
    assert not np.any(np.isnan(s.data))
    assert not np.any(np.isinf(s.data))
    assert np.allclose(s.data, [1.0, 1.0])


def test_sigmoid_stability_large_negative():
    x = Tensor([-500.0, -1000.0])
    s = x.sigmoid()
    assert not np.any(np.isnan(s.data))
    assert np.allclose(s.data, [0.0, 0.0])


def test_exp_overflow():
    x = Tensor([1000.0])
    e = x.exp()
    assert np.isinf(e.data) or e.data > 1e300


def test_log_no_zero():
    with pytest.raises(ValueError):
        Tensor([0.0]).log()


def test_log_no_negative():
    with pytest.raises(ValueError):
        Tensor([-1.0]).log()


def test_sqrt_no_negative():
    with pytest.raises(ValueError):
        Tensor([-1.0]).sqrt()


def test_softmax_gradient_stability():
    x = Tensor([100.0, 101.0, 102.0], requires_grad=True)
    sm = x.softmax()
    loss = sm.sum()
    loss.backward()
    assert not np.any(np.isnan(x.grad))
    assert not np.any(np.isinf(x.grad))


def test_sigmoid_gradient_stability():
    x = Tensor([500.0, -500.0], requires_grad=True)
    s = x.sigmoid()
    loss = s.sum()
    loss.backward()
    assert not np.any(np.isnan(x.grad))
    assert not np.any(np.isinf(x.grad))


def test_division_by_zero_small():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([1e-10, 1e-10])
    c = a / b
    assert not np.any(np.isnan(c.data))


def test_float32_compatibility():
    x = Tensor([1.0, 2.0, 3.0])
    assert x.data.dtype == np.float64


def test_mean_of_empty():
    """mean() debe lanzar ValueError en tensor vacío"""
    with pytest.raises(ValueError):
        Tensor([]).mean()


def test_sum_large_tensor():
    x = Tensor(np.ones(10000) * 1e10)
    s = x.sum()
    assert not np.isnan(s.data)
    assert s.data > 0


def test_float32_mode():
    """Test: modo float32 para móvil"""
    from core.tensor import set_dtype, get_dtype
    
    original = get_dtype()
    try:
        set_dtype(np.float32)
        x = Tensor([1.0, 2.0, 3.0])
        assert x.data.dtype == np.float32, f"Esperado float32, es {x.data.dtype}"
    finally:
        set_dtype(original)


if __name__ == "__main__":
    print("🧪 TESTS DE NUMÉRICA")
    print("="*60)
    pytest.main([__file__, "-v", "-s"])
