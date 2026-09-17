"""
Tests de capas (Linear, Sequential, Flatten)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor
from core.layers import Linear, Sequential, Flatten


def test_linear_forward():
    """Test: forward de Linear"""
    layer = Linear(3, 2)
    x = Tensor([[1.0, 2.0, 3.0]])
    y = layer(x)
    
    assert y.shape == (1, 2), f"Shape incorrecta: {y.shape}"


def test_linear_gradients():
    """Test: gradientes de Linear"""
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Debe haber gradiente en x, weight y bias
    assert x.grad is not None, "No hay gradiente en x"
    assert layer.weight.grad is not None, "No hay gradiente en weight"
    assert layer.bias.grad is not None, "No hay gradiente en bias"


def test_linear_parameters():
    """Test: número de parámetros"""
    layer = Linear(3, 4)
    params = layer.parameters()
    
    # 2 parámetros: weight (3x4) y bias (1x4)
    assert len(params) == 2, f"Esperados 2 parámetros, hay {len(params)}"


def test_sequential_forward():
    """Test: forward de Sequential"""
    model = Sequential(
        Linear(2, 4),
        Linear(4, 1)
    )
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    
    assert y.shape == (1, 1), f"Shape incorrecta: {y.shape}"


def test_sequential_parameters():
    """Test: parámetros de Sequential"""
    model = Sequential(
        Linear(2, 4),
        Linear(4, 1)
    )
    params = model.parameters()
    
    # 2 capas × 2 parámetros = 4
    assert len(params) == 4, f"Esperados 4 parámetros, hay {len(params)}"


def test_flatten():
    """Test: Flatten"""
    flatten = Flatten()
    x = Tensor(np.random.randn(2, 3, 4, 5))
    y = flatten(x)
    
    assert y.shape == (2, 60), f"Shape incorrecta: {y.shape}"


def test_xor_learns():
    """Test: ¿Puede una red aprender XOR?"""
    from core.activations import ReLU, Sigmoid
    from core.losses import MSELoss
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    
    x_t = Tensor(X)
    y_t = Tensor(y)
    
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid()
    )
    
    loss_fn = MSELoss()
    lr = 0.5
    
    losses = []
    for epoch in range(2000):
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        losses.append(loss.data)
        
        for param in model.parameters():
            if param.grad is not None:
                param.data -= lr * param.grad
        
        model.zero_grad()
    
    # El loss debe bajar
    assert losses[-1] < losses[0], "El loss no bajó"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
