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
    """Brain no se puede instanciar directamente"""
    with pytest.raises(TypeError):
        Brain()


def test_mlp_creation():
    """MLP se crea correctamente"""
    model = MLP([2, 4, 1])
    assert model.layer_sizes == [2, 4, 1]
    assert not model.is_trained()


def test_mlp_forward():
    """MLP hace forward correctamente"""
    model = MLP([2, 4, 1])
    x = Tensor([[0.0, 1.0]])
    y = model.forward(x)
    assert y.shape == (1, 1)


def test_mlp_predict():
    """MLP hace predict sin gradientes"""
    model = MLP([2, 4, 1])
    x = Tensor([[0.0, 1.0]])
    y = model.predict(x)
    assert y.shape == (1, 1)


def test_mlp_parameters():
    """MLP devuelve parámetros"""
    model = MLP([2, 4, 1])
    params = model.parameters()
    assert len(params) == 4  # 2 capas × 2 parámetros


def test_mlp_num_parameters():
    """Cuenta correcta de parámetros"""
    model = MLP([2, 4, 1])
    # Capa 1: 2*4 + 4 = 12
    # Capa 2: 4*1 + 1 = 5
    # Total: 17
    assert model.num_parameters() == 17


def test_mlp_train_xor():
    """MLP puede aprender XOR usando train()"""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    
    model = MLP([2, 8, 1])
    
    dataset = [(Tensor(X), Tensor(y))]
    history = model.train(dataset, epochs=2000, lr=0.05, verbose=False)
    
    assert model.is_trained()
    assert len(history) == 2000
    assert history[-1] < history[0], "El loss no bajó"
    
    # Verificar predicciones
    pred = model.predict(Tensor(X)).data
    assert pred[0, 0] < 0.3, f"[0,0] debería ser ~0, es {pred[0,0]}"
    assert pred[1, 0] > 0.7, f"[0,1] debería ser ~1, es {pred[1,0]}"
    assert pred[2, 0] > 0.7, f"[1,0] debería ser ~1, es {pred[2,0]}"
    assert pred[3, 0] < 0.3, f"[1,1] debería ser ~0, es {pred[3,0]}"


def test_mlp_save_load():
    """MLP se guarda y carga correctamente"""
    model = MLP([2, 4, 1])
    
    # Entrenar un poco
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    model.train([(Tensor(X), Tensor(y))], epochs=100, lr=0.1, verbose=False)
    
    # Predicción antes
    pred_before = model.predict(Tensor(X)).data.copy()
    
    # Guardar
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name
    
    model.save(path)
    
    # Crear nuevo modelo y cargar
    model2 = MLP([2, 4, 1])
    model2.load(path)
    
    # Predicción después
    pred_after = model2.predict(Tensor(X)).data
    
    assert np.allclose(pred_before, pred_after), "Las predicciones deben ser idénticas"
    
    os.unlink(path)


def test_mlp_summary():
    """Summary muestra información"""
    model = MLP([2, 4, 1])
    model.summary()
    # Si no falla, OK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
