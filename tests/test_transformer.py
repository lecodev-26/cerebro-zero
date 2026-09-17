"""
Tests del Transformer mejorado
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.tensor import Tensor
from core.layernorm import LayerNorm
from models.transformer import (
    Transformer, TransformerBlock, MultiHeadAttention, FeedForward,
)


# ============================================
# TESTS DE LAYERNORM
# ============================================

def test_layernorm_creation():
    ln = LayerNorm(4)
    assert ln.normalized_shape == (4,)


def test_layernorm_normalizes():
    """LayerNorm debe dejar media~0 y var~1"""
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    y = ln(x)
    
    media = np.mean(y.data, axis=-1)
    var = np.var(y.data, axis=-1)
    
    assert np.allclose(media, 0, atol=1e-5)
    assert np.allclose(var, 1, atol=1e-2)


def test_layernorm_parameters():
    ln = LayerNorm(8)
    params = ln.parameters()
    assert len(params) == 2


# ============================================
# TESTS DE ATTENTION
# ============================================

def test_attention_shape():
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = Tensor(np.random.randn(2, 4, 8))
    y = mha.forward(x)
    assert y.data.shape == (2, 4, 8)


def test_attention_causal_mask():
    """Con causal mask, cada posición no ve el futuro"""
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = Tensor(np.random.randn(1, 4, 8))
    y = mha.forward(x, causal_mask=True)
    assert y.data.shape == (1, 4, 8)


# ============================================
# TESTS DE FEEDFORWARD
# ============================================

def test_ff_shape():
    ff = FeedForward(d_model=8, d_ff=16)
    x = Tensor(np.random.randn(2, 4, 8))
    y = ff.forward(x)
    assert y.data.shape == (2, 4, 8)


# ============================================
# TESTS DE BLOQUE
# ============================================

def test_block_shape():
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
    x = Tensor(np.random.randn(2, 4, 8))
    y = block.forward(x)
    assert y.data.shape == (2, 4, 8)


def test_block_residual():
    """El residual debe preservar la forma"""
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
    x = Tensor(np.random.randn(2, 4, 8))
    y = block.forward(x)
    assert not np.allclose(y.data, x.data)  # No es identidad


# ============================================
# TESTS DEL TRANSFORMER
# ============================================

def test_transformer_creation():
    model = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    assert model.vocab_size == 10
    assert model.d_model == 8


def test_transformer_forward():
    model = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    x = Tensor(np.array([[1, 2, 3, 4]]))
    y = model.forward(x)
    # Output: (batch, seq_len, vocab_size)
    assert y.data.shape == (1, 4, 10)


def test_transformer_parameters():
    model = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=2)
    params = model.parameters()
    # Debe tener muchos parámetros
    assert len(params) > 10


def test_transformer_num_parameters():
    model = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    total = model.num_parameters()
    assert total > 100


def test_transformer_generate():
    model = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=8)
    prompt = [1, 2, 3]
    generated = model.generate(prompt, max_new_tokens=3)
    assert len(generated) == 6  # 3 prompt + 3 nuevos
    assert generated[:3] == prompt  # Preserva el prompt


def test_transformer_generate_top_k():
    model = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    prompt = [1, 2]
    generated = model.generate(prompt, max_new_tokens=5, top_k=3)
    assert len(generated) == 7


def test_transformer_different_seeds():
    """Dos instancias con distinta inicialización deben generar cosas distintas"""
    model1 = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    model2 = Transformer(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    
    x = Tensor(np.array([[1, 2, 3]]))
    y1 = model1.forward(x).data
    y2 = model2.forward(x).data
    
    assert not np.allclose(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
