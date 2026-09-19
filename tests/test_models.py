"""
Tests de modelos completos
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def test_xor_end_to_end():
    """Test: XOR de principio a fin"""
    from core.tensor import Tensor
    from core.layers import Linear, Sequential
    from core.activations import ReLU, Sigmoid
    from core.losses import MSELoss
    from core.optimizers import Adam
    
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
    optimizer = Adam(model.parameters(), lr=0.05)
    
    for epoch in range(3000):
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Verificar predicciones
    pred = model(x_t).data
    assert pred[0, 0] < 0.3, f"[0,0] debería ser ~0, es {pred[0,0]}"
    assert pred[1, 0] > 0.7, f"[0,1] debería ser ~1, es {pred[1,0]}"
    assert pred[2, 0] > 0.7, f"[1,0] debería ser ~1, es {pred[2,0]}"
    assert pred[3, 0] < 0.3, f"[1,1] debería ser ~0, es {pred[3,0]}"


def test_save_load():
    """Test: guardar y cargar modelo (formato seguro JSON+NPY, sin pickle)"""
    from core.tensor import Tensor
    from core.layers import Linear, Sequential
    from core.activations import ReLU, Sigmoid
    from core.serialization import safe_save_dict, safe_load_dict
    import tempfile

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    )

    x = Tensor([[1.0, 2.0]])
    y_before = model(x).data.copy()

    weights = {
        'layer_0_weight': model.layers[0].weight.data,
        'layer_0_bias': model.layers[0].bias.data,
        'layer_2_weight': model.layers[2].weight.data,
        'layer_2_bias': model.layers[2].bias.data,
    }

    # Guardar con serialization segura (crea un directorio)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "modelo_test")
        safe_save_dict(weights, save_path)
        loaded = safe_load_dict(save_path)

    model2 = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    )

    model2.layers[0].weight.data = loaded['layer_0_weight']
    model2.layers[0].bias.data = loaded['layer_0_bias']
    model2.layers[2].weight.data = loaded['layer_2_weight']
    model2.layers[2].bias.data = loaded['layer_2_bias']

    y_after = model2(x).data
    assert np.allclose(y_before, y_after)


def test_gradient_flow():
    """Test: los gradientes fluyen a todas las capas"""
    from core.tensor import Tensor
    from core.layers import Linear, Sequential
    from core.activations import ReLU, Sigmoid
    from core.losses import MSELoss
    
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    )
    
    x = Tensor([[1.0, 2.0]])
    y = Tensor([[1.0]])
    
    pred = model(x)
    loss = MSELoss()(pred, y)
    loss.backward()
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            assert layer.weight.grad is not None, f"Capa {i} sin gradiente en weight"
            assert layer.bias.grad is not None, f"Capa {i} sin gradiente en bias"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
