"""
Fixtures compartidas para todos los tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor, tensor, zeros, ones, randn


@pytest.fixture
def xor_data():
    """Datos XOR para tests"""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    return X, y


@pytest.fixture
def simple_tensor():
    """Tensor simple para tests"""
    return Tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def random_weights():
    """Pesos aleatorios para capas"""
    return np.random.randn(2, 4) * 0.1


@pytest.fixture
def mnist_sample():
    """Muestra de MNIST (datos pequeños)"""
    try:
        X = np.load("modelos_guardados/X_train.npy")[:100]
        y = np.load("modelos_guardados/y_train.npy")[:100]
        return X, y
    except:
        # Si no hay MNIST, usar datos aleatorios
        X = np.random.rand(100, 784)
        y = np.random.randint(0, 10, 100)
        return X, y


@pytest.fixture
def sample_texts():
    """Textos de ejemplo para tests de lenguaje"""
    return [
        "hola mundo",
        "cerebro zero aprende",
        "inteligencia artificial",
        "python y numpy",
    ]
