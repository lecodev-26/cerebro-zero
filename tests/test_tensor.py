"""
Tests básicos del Tensor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor, tensor, zeros, ones, randn


def test_creacion():
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert np.allclose(t.data, [1, 2, 3])
    
    z = zeros((2, 3))
    assert z.shape == (2, 3)
    assert np.allclose(z.data, 0)
    
    o = ones((2, 3))
    assert np.allclose(o.data, 1)
    
    r = randn(2, 3)
    assert r.shape == (2, 3)


def test_operaciones_basicas():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    assert np.allclose((a + b).data, [5, 7, 9])
    assert np.allclose((a - b).data, [-3, -3, -3])
    assert np.allclose((a * b).data, [4, 10, 18])
    assert np.allclose((b / a).data, [4, 2.5, 2])


def test_matmul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a.matmul(b)
    
    esperado = np.array([[19, 22], [43, 50]])
    assert np.allclose(c.data, esperado)


def test_reducciones():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert a.sum().data == 21
    assert np.allclose(a.sum(axis=0).data, [5, 7, 9])
    assert np.allclose(a.sum(axis=1).data, [6, 15])
    assert a.mean().data == 3.5
    assert np.allclose(a.mean(axis=0).data, [2.5, 3.5, 4.5])


def test_activaciones():
    a = Tensor([-1.0, 0.0, 1.0])
    assert np.allclose(a.relu().data, [0, 0, 1])
    
    sig = Tensor([0.0]).sigmoid()
    assert np.allclose(sig.data, 0.5)
    
    tanh = Tensor([0.0]).tanh()
    assert np.allclose(tanh.data, 0.0)
    
    sm = Tensor([1.0, 2.0, 3.0]).softmax()
    assert np.allclose(np.sum(sm.data), 1.0)


def test_funciones_matematicas():
    a = Tensor([1.0, 2.0, 3.0])
    
    assert np.allclose(a.exp().data, np.exp([1.0, 2.0, 3.0]))
    assert np.allclose(a.log().data, np.log([1.0, 2.0, 3.0]))
    assert np.allclose(a.sqrt().data, np.sqrt([1.0, 2.0, 3.0]))
    assert np.allclose(a.pow(2).data, [1.0, 4.0, 9.0])


def test_log_dominio():
    """log(-1) y log(0) deben dar ValueError"""
    with pytest.raises(ValueError):
        Tensor([-1.0]).log()
    
    with pytest.raises(ValueError):
        Tensor([0.0]).log()


def test_sqrt_dominio():
    """sqrt(-1) debe dar ValueError"""
    with pytest.raises(ValueError):
        Tensor([-1.0]).sqrt()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
