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
    """SGD actualiza pesos"""
    layer = Linear(2, 1)
    initial_weight = layer.weight.data.copy()
    
    layer.weight.grad = np.ones_like(layer.weight.data)
    layer.bias.grad = np.ones_like(layer.bias.data)
    
    optimizer = SGD(layer.parameters(), lr=0.1)
    optimizer.step()
    
    assert not np.allclose(layer.weight.data, initial_weight)


def test_adam_step():
    """Adam actualiza pesos"""
    layer = Linear(2, 1)
    initial_weight = layer.weight.data.copy()
    
    layer.weight.grad = np.ones_like(layer.weight.data)
    layer.bias.grad = np.ones_like(layer.bias.data)
    
    optimizer = Adam(layer.parameters(), lr=0.1)
    optimizer.step()
    
    assert not np.allclose(layer.weight.data, initial_weight)


def test_adam_converges():
    """Adam reduce el loss en XOR"""
    from core.layers import Sequential
    from core.activations import ReLU, Sigmoid
    from core.losses import MSELoss
    
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    
    x_t = Tensor(X)
    y_t = Tensor(y)
    
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 1),
        Sigmoid()
    )
    
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.1)
    
    initial_loss = None
    final_loss = None
    
    for epoch in range(5000):
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        
        if epoch == 0:
            initial_loss = float(loss.data)
        final_loss = float(loss.data)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # El loss debe bajar significativamente (al menos 30%)
    mejora = (initial_loss - final_loss) / initial_loss
    assert mejora > 0.3, (
        f"Adam no convergió lo suficiente: "
        f"{initial_loss:.4f} → {final_loss:.4f} (mejora={mejora*100:.1f}%)"
    )


def test_sgd_momentum():
    """SGD con momentum"""
    layer = Linear(2, 1)
    initial = layer.weight.data.copy()
    
    layer.weight.grad = np.ones_like(layer.weight.data)
    layer.bias.grad = np.ones_like(layer.bias.data)
    
    optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.9)
    optimizer.step()
    
    assert not np.allclose(layer.weight.data, initial)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
