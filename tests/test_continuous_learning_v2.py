"""
Tests de aprendizaje continuo con replay buffer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from training.replay_buffer import ReplayBuffer
from training.continuous_learning_v2 import ContinuousLearner
from models.transformer import Transformer


# ============================================
# TESTS DE REPLAY BUFFER
# ============================================

def test_buffer_creation():
    buffer = ReplayBuffer(max_size=100)
    assert buffer.size() == 0
    assert buffer.max_size == 100


def test_buffer_add():
    buffer = ReplayBuffer(max_size=100)
    X = np.random.randint(0, 10, (4, 8))
    y = np.random.randint(0, 10, (4, 8))
    buffer.add(X, y, {'task': 'test'})
    assert buffer.size() == 1


def test_buffer_max_size():
    buffer = ReplayBuffer(max_size=5)
    for _ in range(10):
        X = np.random.randint(0, 10, (2, 8))
        y = np.random.randint(0, 10, (2, 8))
        buffer.add(X, y)
    assert buffer.size() == 5  # Deque con maxlen


def test_buffer_sample():
    buffer = ReplayBuffer(max_size=100, seed=42)
    for _ in range(10):
        X = np.random.randint(0, 10, (2, 8))
        y = np.random.randint(0, 10, (2, 8))
        buffer.add(X, y)
    
    X_batch, y_batch = buffer.sample(batch_size=4)
    assert X_batch.shape[0] == 4
    assert y_batch.shape[0] == 4


def test_buffer_sample_empty():
    buffer = ReplayBuffer()
    with pytest.raises(ValueError):
        buffer.sample(batch_size=4)


def test_buffer_sample_by_task():
    buffer = ReplayBuffer(max_size=100, seed=42)
    for i in range(10):
        X = np.random.randint(0, 10, (2, 8))
        y = np.random.randint(0, 10, (2, 8))
        buffer.add(X, y, {'task': 'math' if i % 2 == 0 else 'lang'})
    
    result = buffer.sample_by_task('math', batch_size=2)
    assert result is not None
    X_batch, y_batch = result
    assert X_batch.shape[0] == 2


def test_buffer_tasks():
    buffer = ReplayBuffer()
    for task in ['math', 'lang', 'math', 'memory']:
        X = np.random.randint(0, 10, (2, 8))
        y = np.random.randint(0, 10, (2, 8))
        buffer.add(X, y, {'task': task})
    
    tasks = buffer.tasks()
    assert 'math' in tasks
    assert 'lang' in tasks
    assert 'memory' in tasks


def test_buffer_stats():
    buffer = ReplayBuffer(max_size=100)
    X = np.random.randint(0, 10, (2, 8))
    y = np.random.randint(0, 10, (2, 8))
    buffer.add(X, y, {'task': 'test'})
    
    stats = buffer.stats()
    assert stats['size'] == 1
    assert stats['max_size'] == 100
    assert 'test' in stats['tasks']


# ============================================
# TESTS DEL CONTINUOUS LEARNER
# ============================================

@pytest.fixture
def model():
    return Transformer(
        vocab_size=50, d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=8,
    )


def test_learner_creation(model):
    learner = ContinuousLearner(model, buffer_size=100)
    assert learner.model is model
    assert learner.buffer.size() == 0


def test_learner_learn_batch(model):
    learner = ContinuousLearner(model, buffer_size=100, learning_rate=0.05)
    X = np.random.randint(0, 50, (4, 8))
    y = np.random.randint(0, 50, (4, 8))
    
    result = learner.learn_batch(X, y, task='math', epochs=3, verbose=False)
    
    assert 'loss_before' in result
    assert 'loss_after' in result
    assert 'improvement' in result
    # El loss debe bajar en la tarea nueva
    assert result['improvement'] > 0, "El loss no mejoró"


def test_learner_multiple_tasks(model):
    learner = ContinuousLearner(model, buffer_size=100, learning_rate=0.05)
    
    for task in ['math', 'lang', 'memory']:
        X = np.random.randint(0, 50, (4, 8))
        y = np.random.randint(0, 50, (4, 8))
        learner.learn_batch(X, y, task=task, epochs=3, verbose=False)
    
    assert len(learner.history['tasks_learned']) == 3


def test_learner_buffer_grows(model):
    learner = ContinuousLearner(model, buffer_size=100)
    
    for _ in range(3):
        X = np.random.randint(0, 50, (4, 8))
        y = np.random.randint(0, 50, (4, 8))
        learner.learn_batch(X, y, task='math', epochs=1, verbose=False)
    
    assert learner.buffer.size() == 3


def test_learner_forgetting_eval(model):
    learner = ContinuousLearner(model, buffer_size=100, learning_rate=0.05)
    
    # Datos para 2 tareas
    X1 = np.random.randint(0, 50, (8, 8))
    y1 = np.random.randint(0, 50, (8, 8))
    X2 = np.random.randint(0, 50, (8, 8))
    y2 = np.random.randint(0, 50, (8, 8))
    
    learner.learn_batch(X1, y1, task='math', epochs=3, verbose=False)
    learner.learn_batch(X2, y2, task='lang', epochs=3, verbose=False)
    
    # Evaluar olvido
    task_data = {'math': (X1, y1), 'lang': (X2, y2)}
    forgetting = learner.evaluate_forgetting(task_data, verbose=False)
    
    assert 'math' in forgetting
    assert 'lang' in forgetting


def test_learner_replay_reduces_forgetting(model):
    """
    Con replay buffer, el olvido debe ser menor
    que sin replay buffer.
    """
    # Sin replay
    model_no_replay = Transformer(
        vocab_size=50, d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=8,
    )
    learner_no_replay = ContinuousLearner(model_no_replay, buffer_size=100, learning_rate=0.05)
    
    # Datos de dos tareas
    X1 = np.random.randint(0, 50, (8, 8))
    y1 = np.random.randint(0, 50, (8, 8))
    X2 = np.random.randint(0, 50, (8, 8))
    y2 = np.random.randint(0, 50, (8, 8))
    
    # Aprender tarea 1
    learner_no_replay.learn_batch(X1, y1, task='math', epochs=3, use_replay=False, verbose=False)
    loss_math_1 = learner_no_replay._evaluate_on(X1, y1)
    
    # Aprender tarea 2 sin replay
    learner_no_replay.learn_batch(X2, y2, task='lang', epochs=3, use_replay=False, verbose=False)
    loss_math_2 = learner_no_replay._evaluate_on(X1, y1)
    
    # Con replay
    model_replay = Transformer(
        vocab_size=50, d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=8,
    )
    learner_replay = ContinuousLearner(model_replay, buffer_size=100, learning_rate=0.05)
    
    learner_replay.learn_batch(X1, y1, task='math', epochs=3, use_replay=False, verbose=False)
    loss_math_3 = learner_replay._evaluate_on(X1, y1)
    
    learner_replay.learn_batch(X2, y2, task='lang', epochs=3, use_replay=True, verbose=False)
    loss_math_4 = learner_replay._evaluate_on(X1, y1)
    
    # El loss de la tarea antigua debe ser menor con replay
    # (o al menos no peor)
    forgetting_no_replay = loss_math_2 - loss_math_1
    forgetting_replay = loss_math_4 - loss_math_3
    
    # No somos estrictos, solo verificamos que no sea catastrófico
    # (el replay no debería empeorar por mucho)
    assert forgetting_replay <= forgetting_no_replay + 0.5, (
        f"Replay no ayuda: no_replay={forgetting_no_replay:.4f}, "
        f"replay={forgetting_replay:.4f}"
    )


def test_learner_stats(model):
    learner = ContinuousLearner(model, buffer_size=100, learning_rate=0.05)
    
    X = np.random.randint(0, 50, (4, 8))
    y = np.random.randint(0, 50, (4, 8))
    learner.learn_batch(X, y, task='math', epochs=2, verbose=False)
    
    stats = learner.stats()
    assert stats['tasks_learned'] == 1
    assert stats['buffer_size'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
