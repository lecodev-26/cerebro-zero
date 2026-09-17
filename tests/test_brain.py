"""
Tests de la interfaz Brain
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile
from core.tensor import Tensor
from models.brain import Brain
from models.mlp import MLP


def test_brain_is_abstract():
    with pytest.raises(TypeError):
        Brain()


def test_mlp_creation():
    model = MLP([2, 4, 1])
    assert model.layer_sizes == [2, 4, 1]
    assert not model.is_trained()


def test_mlp_forward():
    model = MLP([2, 4, 1])
    x = Tensor([[0.0, 1.0]])
    y = model.forward(x)
    assert y.shape == (1, 1)


def test_mlp_predict():
    model = MLP([2, 4, 1])
    x = Tensor([[0.0, 1.0]])
    y = model.predict(x)
    assert y.shape == (1, 1)


def test_mlp_parameters():
    model = MLP([2, 4, 1])
    params = model.parameters()
    assert len(params) == 4


def test_mlp_num_parameters():
    model = MLP([2, 4, 1])
    # Capa 1: 2*4 + 4 = 12
    # Capa 2: 4*1 + 1 = 5
    # Total: 17
    assert model.num_parameters() == 17


def test_mlp_train_xor():
    """MLP puede aprender XOR usando train()"""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    
    # Red más grande + lr más alto
    model = MLP([2, 16, 1])
    
    dataset = [(Tensor(X), Tensor(y))]
    history = model.train(dataset, epochs=5000, lr=0.1, verbose=False)
    
    assert model.is_trained()
    assert history[-1] < history[0], "El loss no bajó"
    
    # Verificar predicciones
    pred = model.predict(Tensor(X)).data
    # Ser más permisivos pero que separen las clases
    assert pred[0, 0] < 0.3, f"[0,0] debería ser ~0, es {pred[0,0]}"
    assert pred[1, 0] > 0.7, f"[0,1] debería ser ~1, es {pred[1,0]}"
    assert pred[2, 0] > 0.7, f"[1,0] debería ser ~1, es {pred[2,0]}"
    assert pred[3, 0] < 0.3, f"[1,1] debería ser ~0, es {pred[3,0]}"


def test_mlp_save_load():
    model = MLP([2, 4, 1])
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    model.train([(Tensor(X), Tensor(y))], epochs=100, lr=0.1, verbose=False)
    
    pred_before = model.predict(Tensor(X)).data.copy()
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name
    
    model.save(path)
    
    model2 = MLP([2, 4, 1])
    model2.load(path)
    
    pred_after = model2.predict(Tensor(X)).data
    
    assert np.allclose(pred_before, pred_after)
    
    os.unlink(path)


def test_mlp_summary():
    model = MLP([2, 4, 1])
    model.summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
