"""
Tests del entrenador de Language Model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from language.tokenizer import Tokenizer
from language.dataset import TextDataset
from models.transformer import Transformer
from training.lm_trainer import LMTrainer


@pytest.fixture
def setup():
    """Configurar tokenizer y datasets"""
    all_texts = []
    for split in ['train', 'validation', 'test']:
        path = f"datasets/{split}.txt"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
    
    tok = Tokenizer()
    tok.build_vocab(all_texts, vocab_size=100)
    
    train_ds = TextDataset("datasets/train.txt", tok, context_len=8)
    val_ds = TextDataset("datasets/validation.txt", tok, context_len=8)
    
    return tok, train_ds, val_ds


@pytest.fixture
def model(setup):
    tok, _, _ = setup
    return Transformer(
        vocab_size=tok.vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=8,
    )


# ============================================
# TESTS DEL TRAINER
# ============================================

def test_trainer_creation(model):
    trainer = LMTrainer(model, learning_rate=0.01)
    assert trainer.model is model
    assert trainer.lr == 0.01


def test_cross_entropy_loss(model):
    trainer = LMTrainer(model)
    
    # logits random, targets
    logits = np.random.randn(2, 4, 10)
    targets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    
    loss = trainer.cross_entropy_loss(logits, targets)
    assert loss > 0
    assert not np.isnan(loss)


def test_compute_gradients(model):
    trainer = LMTrainer(model)
    
    logits = np.random.randn(2, 4, 10)
    targets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    
    grads = trainer.compute_logit_gradients(logits, targets)
    assert grads.shape == logits.shape


def test_train_step(model, setup):
    tok, train_ds, _ = setup
    trainer = LMTrainer(model, learning_rate=0.01)
    
    X, y = train_ds.get_batch(batch_size=2)
    loss = trainer.train_step(X, y)
    
    assert loss > 0
    assert not np.isnan(loss)


def test_evaluate(model, setup):
    tok, _, val_ds = setup
    trainer = LMTrainer(model)
    
    metrics = trainer.evaluate(val_ds, batch_size=2, num_batches=2)
    assert 'loss' in metrics
    assert 'ppl' in metrics
    assert metrics['ppl'] > 0


def test_train_reduces_loss(model, setup):
    """El loss debe bajar después de varias épocas"""
    tok, train_ds, val_ds = setup
    trainer = LMTrainer(model, learning_rate=0.05)
    
    history = trainer.train(
        train_ds, val_ds,
        epochs=10,
        batch_size=4,
        batches_per_epoch=10,
        verbose=False,
    )
    
    # Debe haber reducción significativa
    assert len(history['train_loss']) == 10
    initial = history['train_loss'][0]
    final = min(history['train_loss'])  # Mejor de todas las épocas
    assert final < initial * 0.95, (
        f"El loss no bajó lo suficiente: {initial:.4f} → {final:.4f}"
    )


def test_trainer_history(model, setup):
    tok, train_ds, _ = setup
    trainer = LMTrainer(model)
    
    history = trainer.train(
        train_ds, epochs=3, batch_size=2, batches_per_epoch=3, verbose=False
    )
    
    assert 'train_loss' in history
    assert 'train_ppl' in history
    assert len(history['train_loss']) == 3


def test_trainer_save_load(model, setup, tmp_path):
    tok, train_ds, _ = setup
    trainer = LMTrainer(model)
    
    trainer.train(train_ds, epochs=2, batch_size=2, batches_per_epoch=2, verbose=False)
    
    path = str(tmp_path / "test_model.pkl")
    trainer.save(path)
    
    assert os.path.exists(path)
    
    # Cargar en nuevo trainer
    model2 = Transformer(
        vocab_size=tok.vocab_size, d_model=16, num_heads=2,
        d_ff=32, num_layers=1, max_len=8,
    )
    trainer2 = LMTrainer(model2)
    trainer2.load(path)
    
    # Verificar que el histórico se carga
    assert len(trainer2.history['train_loss']) == 2


def test_trainer_best_epoch(model, setup):
    tok, train_ds, val_ds = setup
    trainer = LMTrainer(model, learning_rate=0.05)
    
    trainer.train(
        train_ds, val_ds,
        epochs=3,
        batch_size=4,
        batches_per_epoch=3,
        verbose=False,
        eval_every=1,
    )
    
    assert trainer.best_epoch >= 0
    assert trainer.best_val_loss < float('inf')


def test_perplexity_decreases(model, setup):
    """La perplejidad debe bajar con el entrenamiento"""
    tok, train_ds, val_ds = setup
    trainer = LMTrainer(model, learning_rate=0.05)
    
    history = trainer.train(
        train_ds, val_ds,
        epochs=10,
        batch_size=4,
        batches_per_epoch=10,
        verbose=False,
    )
    
    ppl_initial = history['train_ppl'][0]
    ppl_best = min(history['train_ppl'])
    
    assert ppl_best < ppl_initial * 0.95, (
        f"La perplejidad no bajó: {ppl_initial:.2f} → mejor={ppl_best:.2f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
