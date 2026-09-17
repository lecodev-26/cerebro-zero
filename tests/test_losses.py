"""
Tests de funciones de pérdida
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor
from core.losses import MSELoss, BCELoss, CrossEntropyLoss


def test_mse_zero():
    """Test: MSE entre valores iguales = 0"""
    loss_fn = MSELoss()
    pred = Tensor([[1.0, 2.0]])
    target = Tensor([[1.0, 2.0]])
    loss = loss_fn(pred, target)
    
    assert np.isclose(loss.data, 0.0), f"MSE debe ser 0, es {loss.data}"


def test_mse_positive():
    """Test: MSE siempre positivo"""
    loss_fn = MSELoss()
    pred = Tensor([[1.0, 2.0]])
    target = Tensor([[0.0, 0.0]])
    loss = loss_fn(pred, target)
    
    assert loss.data > 0


def test_mse_gradients():
    """Test: gradientes de MSE"""
    loss_fn = MSELoss()
    pred = Tensor([[1.0, 2.0]], requires_grad=True)
    target = Tensor([[0.0, 0.0]])
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None


def test_bce_gradients():
    """Test: gradientes de BCE"""
    loss_fn = BCELoss()
    pred = Tensor([[0.7, 0.3]], requires_grad=True)
    target = Tensor([[1.0, 0.0]])
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None
    assert loss.data > 0


def test_cross_entropy():
    """Test: CrossEntropy"""
    loss_fn = CrossEntropyLoss()
    logits = Tensor([[2.0, 1.0, 0.1]])
    target = Tensor([0])
    loss = loss_fn(logits, target)
    
    assert loss.data > 0


def test_cross_entropy_perfect():
    """Test: CrossEntropy con predicción perfecta"""
    loss_fn = CrossEntropyLoss()
    # Logits muy altos para clase 0
    logits = Tensor([[100.0, 0.0, 0.0]])
    target = Tensor([0])
    loss = loss_fn(logits, target)
    
    # El loss debe ser cercano a 0
    assert loss.data < 0.1, f"Loss debe ser pequeño, es {loss.data}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
