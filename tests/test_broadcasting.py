"""
Tests exhaustivos de broadcasting en autograd
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor


def numerical_gradient(func, x, eps=1e-6):
    """Gradiente numérico con diferencias finitas centrales"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + eps
        f_plus = func(x)
        x[idx] = old_val - eps
        f_minus = func(x)
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x[idx] = old_val
        it.iternext()
    return grad


def check_broadcast_gradients(name, op_func, *inputs, tol=1e-4):
    """Verifica gradientes con broadcasting"""
    tensors = [Tensor(inp.copy(), requires_grad=True) for inp in inputs]
    output = op_func(*tensors)
    output.backward()
    
    analytic_grads = [t.grad.copy() for t in tensors]
    
    numeric_grads = []
    for i, t in enumerate(tensors):
        def loss_fn(x):
            new_inputs = [inp.copy() for inp in inputs]
            new_inputs[i] = x
            new_tensors = [Tensor(inp, requires_grad=False) for inp in new_inputs]
            out = op_func(*new_tensors)
            return np.sum(out.data)
        
        ng = numerical_gradient(loss_fn, t.data.copy())
        numeric_grads.append(ng)
    
    for i, (analytic, numeric) in enumerate(zip(analytic_grads, numeric_grads)):
        diff = np.max(np.abs(analytic - numeric))
        assert diff < tol, f"{name} (input {i}): diff={diff}, analytic shape={analytic.shape}, numeric shape={numeric.shape}"
    
    print(f"   ✅ {name}")


def test_broadcast_scalar_add():
    """a (2,3) + b (1,)"""
    check_broadcast_gradients("(2,3) + (1,)", 
        lambda a, b: a + b,
        np.random.randn(2, 3),
        np.random.randn(1))


def test_broadcast_scalar_mul():
    """a (2,3) * b (1,)"""
    check_broadcast_gradients("(2,3) * (1,)",
        lambda a, b: a * b,
        np.random.randn(2, 3),
        np.random.randn(1))


def test_broadcast_row():
    """a (3,4) + b (1,4)"""
    check_broadcast_gradients("(3,4) + (1,4)",
        lambda a, b: a + b,
        np.random.randn(3, 4),
        np.random.randn(1, 4))


def test_broadcast_col():
    """a (3,4) + b (3,1)"""
    check_broadcast_gradients("(3,4) + (3,1)",
        lambda a, b: a + b,
        np.random.randn(3, 4),
        np.random.randn(3, 1))


def test_broadcast_2d_3d():
    """a (2,3,4) + b (3,4)"""
    check_broadcast_gradients("(2,3,4) + (3,4)",
        lambda a, b: a + b,
        np.random.randn(2, 3, 4),
        np.random.randn(3, 4))


def test_broadcast_batch():
    """a (2,3,4) + b (1,4)"""
    check_broadcast_gradients("(2,3,4) + (1,4)",
        lambda a, b: a + b,
        np.random.randn(2, 3, 4),
        np.random.randn(1, 4))


def test_broadcast_mul_2d():
    """a (3,4) * b (1,4)"""
    check_broadcast_gradients("(3,4) * (1,4)",
        lambda a, b: a * b,
        np.random.randn(3, 4),
        np.random.randn(1, 4))


def test_broadcast_div():
    """a (3,4) / b (1,4)"""
    check_broadcast_gradients("(3,4) / (1,4)",
        lambda a, b: a / b,
        np.random.randn(3, 4),
        np.random.randn(1, 4) + 2)  # +2 para evitar división por cero


def test_broadcast_chain():
    """Cadena con broadcasting"""
    check_broadcast_gradients("(a*b + c).sum() con broadcasting",
        lambda a, b, c: (a * b + c).sum(),
        np.random.randn(3, 4),
        np.random.randn(1, 4),
        np.random.randn(3, 1))


def test_broadcast_neural_layer():
    """Capa lineal con bias (broadcasting)"""
    check_broadcast_gradients("capa lineal: x@w + b",
        lambda x, w, b: (x.matmul(w) + b).sum(),
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(1, 4))


if __name__ == "__main__":
    print("🧪 TESTS DE BROADCASTING")
    print("="*60)
    pytest.main([__file__, "-v", "-s"])
