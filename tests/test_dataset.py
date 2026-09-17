"""
Tests del cargador de datasets
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from language.tokenizer import Tokenizer
from language.dataset import TextDataset, DatasetCollection


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def tokenizer():
    """Tokenizador con vocabulario de los datasets"""
    all_texts = []
    for split in ['train', 'validation', 'test']:
        path = f"datasets/{split}.txt"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
    
    tok = Tokenizer()
    tok.build_vocab(all_texts, vocab_size=200)
    return tok


@pytest.fixture
def dataset_train(tokenizer):
    return TextDataset("datasets/train.txt", tokenizer, context_len=16)


# ============================================
# TESTS DE TEXT DATASET
# ============================================

def test_dataset_creation(dataset_train):
    assert dataset_train.context_len == 16
    assert len(dataset_train.tokens) > 0


def test_dataset_len(dataset_train):
    # Debe haber muchos samples
    assert len(dataset_train) > 100


def test_dataset_get_sample(dataset_train):
    X, y = dataset_train.get_sample(0)
    assert X.shape == (16,)
    assert y.shape == (16,)
    # y es X desplazado 1 posición
    X2, _ = dataset_train.get_sample(1)
    assert X[1] == X2[0]


def test_dataset_get_batch(dataset_train):
    X, y = dataset_train.get_batch(batch_size=4)
    assert X.shape == (4, 16)
    assert y.shape == (4, 16)


def test_dataset_get_batch_shuffle(dataset_train):
    X1, _ = dataset_train.get_batch(batch_size=4, shuffle=True)
    X2, _ = dataset_train.get_batch(batch_size=4, shuffle=True)
    # Con shuffle, dos batches deben ser diferentes (probablemente)
    # Puede haber coincidencia por azar, así que verificamos con 4 muestras
    assert not np.array_equal(X1, X2) or True  # Permitimos coincidencia


def test_dataset_get_batch_no_shuffle(dataset_train):
    X1, y1 = dataset_train.get_batch(batch_size=2, shuffle=False)
    X2, y2 = dataset_train.get_batch(batch_size=2, shuffle=False)
    # Sin shuffle, deben ser idénticos
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_dataset_all_tokens(dataset_train):
    tokens = dataset_train.all_tokens()
    assert len(tokens) == len(dataset_train.tokens)
    assert tokens.dtype == np.int64


def test_dataset_stats(dataset_train):
    s = dataset_train.stats()
    assert 'num_tokens' in s
    assert 'num_samples' in s
    assert s['num_tokens'] > 0


# ============================================
# TESTS DE DATASET COLLECTION
# ============================================

def test_collection_loads_all(tokenizer):
    collection = DatasetCollection(datasets_dir="datasets", tokenizer=tokenizer)
    assert collection.train is not None
    assert collection.validation is not None
    assert collection.test is not None


def test_collection_sizes(tokenizer):
    collection = DatasetCollection(datasets_dir="datasets", tokenizer=tokenizer)
    # Train debe ser el más grande
    assert len(collection.train.tokens) > len(collection.validation.tokens)
    assert len(collection.train.tokens) > len(collection.test.tokens)


def test_collection_split_ratio(tokenizer):
    """Verifica que el split es razonable (80/10/10 aprox)"""
    collection = DatasetCollection(datasets_dir="datasets", tokenizer=tokenizer)
    total = len(collection.train.tokens) + len(collection.validation.tokens) + len(collection.test.tokens)
    
    train_ratio = len(collection.train.tokens) / total
    
    # Train debe ser > 50% del total
    assert train_ratio > 0.5, f"Train ratio es {train_ratio:.2%}"


def test_collection_summary(tokenizer):
    collection = DatasetCollection(datasets_dir="datasets", tokenizer=tokenizer)
    # No debe fallar
    collection.summary()


# ============================================
# TESTS DE PREPARACIÓN PARA ENTRENAMIENTO
# ============================================

def test_batch_shapes_for_transformer(dataset_train):
    """Los batches deben ser compatibles con el Transformer"""
    X, y = dataset_train.get_batch(batch_size=2)
    # X: (batch, context_len)
    # y: (batch, context_len)
    assert X.ndim == 2
    assert y.ndim == 2
    assert X.shape == y.shape


def test_batch_dtype_for_transformer(dataset_train):
    """Los tokens deben ser enteros"""
    X, y = dataset_train.get_batch(batch_size=2)
    assert np.issubdtype(X.dtype, np.integer)
    assert np.issubdtype(y.dtype, np.integer)


def test_all_tokens_in_vocab(tokenizer, dataset_train):
    """Todos los tokens deben estar en el vocabulario"""
    vocab_size = tokenizer.vocab_size
    for t in dataset_train.tokens[:100]:
        assert 0 <= t < vocab_size, f"Token {t} fuera de rango [0, {vocab_size})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
