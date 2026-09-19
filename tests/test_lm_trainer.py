"""
Tests del LMTrainer V3.0 (backprop real)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor
from models.transformer import Transformer
from training.lm_trainer import LMTrainer


@pytest.fixture
def model():
    return Transformer(vocab_size=20, d_model=16, num_heads=2,
                       d_ff=32, num_layers=1, max_len=8)


def test_trainer_creation(model):
    trainer = LMTrainer(model, learning_rate=0.01)
    assert trainer.model is model
    assert trainer.lr == 0.01


def test_cross_entropy_loss(model):
    """Cross entropy con Tensor"""
    trainer = LMTrainer(model)
    
    # Logits como Tensor
    logits_data = np.random.randn(2, 4, 20)
    logits = Tensor(logits_data, requires_grad=True)
    targets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    
    loss = trainer.cross_entropy_loss(logits, targets)
    assert loss.data > 0
    assert not np.isnan(loss.data)


def test_train_step(model):
    trainer = LMTrainer(model, learning_rate=0.01)
    
    X = np.random.randint(0, 20, (2, 8))
    y = np.random.randint(0, 20, (2, 8))
    
    loss = trainer.train_step(X, y)
    assert loss > 0


def test_trainer_history(model):
    trainer = LMTrainer(model)
    
    # Historial vacío al inicio
    assert len(trainer.history['train_loss']) == 0


def test_trainer_save_load(model, tmp_path):
    """Test: guardar y cargar trainer (formato seguro JSON+NPY)"""
    trainer = LMTrainer(model)

    # Sin extensión .pkl → safe_save_dict crea un directorio
    path = str(tmp_path / "test_model")
    trainer.save(path)
    assert os.path.isdir(path)
    assert os.path.exists(os.path.join(path, "metadata.json"))

    # Cargar en un modelo nuevo
    model2 = Transformer(vocab_size=20, d_model=16, num_heads=2,
                        d_ff=32, num_layers=1, max_len=8)
    trainer2 = LMTrainer(model2)
    trainer2.load(path)

    # Verificar que los pesos coinciden
    for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
        assert np.allclose(p1.data, p2.data)


def test_gradient_flow_real(model):
    """Verifica que TODOS los parámetros reciben gradiente con backprop real"""
    X = np.random.randint(0, 20, (2, 8))
    y = np.random.randint(0, 20, (2, 8))
    
    # Forward
    logits = model.forward(Tensor(X))
    loss = LMTrainer(model).cross_entropy_loss(logits, y)
    loss.backward()
    
    # Todos los parámetros deben tener gradiente
    for i, param in enumerate(model.parameters()):
        assert param.grad is not None, f"Parámetro {i} sin gradiente"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
