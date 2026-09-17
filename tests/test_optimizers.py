"""
Tests de optimizadores
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor
from core.layers import Linear
from core.optimizers import SGD, Adam


def test_sgd_step():
    """Test: SGD actualiza pesos"""
    layer = Linear(2, 1)
    initial_weight = layer.weight.data.copy()
    
    # Simular gradiente
    layer.weight.grad = np.ones_like(layer.weight.data)
    layer.bias.grad = np.ones_like(layer.bias.data)
    
    optimizer = SGD(layer.parameters(), lr=0.1)
    optimizer.step()
    
    # Los pesos deben haber cambiado
    assert not np.allclose(layer.weight.data, initial_weight)


def test_adam_step():
    """Test: Adam actualiza pesos"""
    layer = Linear(2, 1)
    initial_weight = layer.weight.data.copy()
    
    layer.weight.grad = np.ones_like(layer.weight.data)
    layer.bias.grad = np.ones_like(layer.bias.data)
    
    optimizer = Adam(layer.parameters(), lr=0.1)
    optimizer.step()
    
    assert not np.allclose(layer.weight.data, initial_weight)


def test_adam_converges():
    """Test: Adam converge en XOR"""
    from core.layers import Sequential
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
    optimizer = Adam(model.parameters(), lr=0.05)
    
    initial_loss = None
    final_loss = None
    
    for epoch in range(3000):
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        
        if epoch == 0:
            initial_loss = loss.data
        final_loss = loss.data
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # El loss debe bajar significativamente
    assert final_loss < initial_loss * 0.5, f"Adam no convergió: {initial_loss} -> {final_loss}"


def test_sgd_momentum():
    """Test: SGD con momentum"""
    layer = Linear(2, 1)
    initial = layer.weight.data.copy()
    
    layer.weight.grad = np.ones_like(layer.weight.data)
    layer.bias.grad = np.ones_like(layer.bias.data)
    
    optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.9)
    optimizer.step()
    
    assert not np.allclose(layer.weight.data, initial)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
