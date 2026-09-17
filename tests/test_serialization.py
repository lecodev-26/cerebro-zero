"""
Tests de persistencia segura
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import json
from core.serialization import (
    safe_save_model, safe_load_model,
    safe_save_dict, safe_load_dict,
    compute_hash, MODEL_FORMAT_VERSION,
)
from models.transformer import Transformer
from models.mlp import MLP


@pytest.fixture
def model():
    return Transformer(vocab_size=10, d_model=8, num_heads=2,
                       d_ff=16, num_layers=1, max_len=4)


@pytest.fixture
def mlp():
    return MLP([2, 4, 1])


# ============================================
# TESTS BÁSICOS
# ============================================

def test_compute_hash():
    h1 = compute_hash(b"hello")
    h2 = compute_hash(b"hello")
    h3 = compute_hash(b"world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA256 hex


def test_save_creates_structure(model, tmp_path):
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    # Debe existir metadata.json, manifest.json, tensors/
    assert os.path.exists(os.path.join(save_path, "metadata.json"))
    assert os.path.exists(os.path.join(save_path, "manifest.json"))
    assert os.path.isdir(os.path.join(save_path, "tensors"))


def test_save_creates_tensors(model, tmp_path):
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    tensors_dir = os.path.join(save_path, "tensors")
    files = sorted(os.listdir(tensors_dir))
    assert len(files) == len(model.parameters())
    
    for f in files:
        assert f.endswith(".npy")


def test_save_metadata_content(model, tmp_path):
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    with open(os.path.join(save_path, "metadata.json")) as f:
        metadata = json.load(f)
    
    assert metadata['format_version'] == MODEL_FORMAT_VERSION
    assert metadata['class'] == 'Transformer'
    assert metadata['num_parameters'] > 0
    assert 'saved_at' in metadata


def test_save_manifest_has_hashes(model, tmp_path):
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    with open(os.path.join(save_path, "manifest.json")) as f:
        manifest = json.load(f)
    
    for tensor in manifest['tensors']:
        assert 'hash_sha256' in tensor
        assert len(tensor['hash_sha256']) == 64
        assert 'shape' in tensor
        assert 'dtype' in tensor


# ============================================
# TESTS DE CARGA
# ============================================

def test_load_restores_model(model, tmp_path):
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    # Cargar en modelo nuevo
    model2 = Transformer(vocab_size=10, d_model=8, num_heads=2,
                        d_ff=16, num_layers=1, max_len=4)
    safe_load_model(model2, save_path)
    
    # Verificar
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert np.array_equal(p1.data, p2.data)


def test_load_mlp(mlp, tmp_path):
    save_path = str(tmp_path / "mlp")
    safe_save_model(mlp, save_path)
    
    mlp2 = MLP([2, 4, 1])
    safe_load_model(mlp2, save_path)
    
    for p1, p2 in zip(mlp.parameters(), mlp2.parameters()):
        assert np.array_equal(p1.data, p2.data)


def test_load_nonexistent_path(model):
    with pytest.raises(FileNotFoundError):
        safe_load_model(model, "/nonexistent/path")


def test_load_wrong_param_count(model, tmp_path):
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    # MLP tiene menos parámetros
    mlp_wrong = MLP([2, 4, 1])
    with pytest.raises(ValueError):
        safe_load_model(mlp_wrong, save_path)


# ============================================
# TESTS DE INTEGRIDAD
# ============================================

def test_integrity_detects_corruption(model, tmp_path):
    """Modificar un tensor debe hacer fallar la carga"""
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    # Corromper un archivo
    tensors_dir = os.path.join(save_path, "tensors")
    target = os.path.join(tensors_dir, "0000.npy")
    data = np.load(target)
    data[0, 0] = 99999
    np.save(target, data)
    
    # Cargar debe fallar por hash
    model2 = Transformer(vocab_size=10, d_model=8, num_heads=2,
                        d_ff=16, num_layers=1, max_len=4)
    with pytest.raises(ValueError, match="Hash no coincide"):
        safe_load_model(model2, save_path)


def test_integrity_can_be_skipped(model, tmp_path):
    """Verificar hashes es opcional"""
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    # Corromper
    tensors_dir = os.path.join(save_path, "tensors")
    target = os.path.join(tensors_dir, "0000.npy")
    data = np.load(target)
    data[0, 0] = 99999
    np.save(target, data)
    
    # Cargar sin verificar → no debe fallar
    model2 = Transformer(vocab_size=10, d_model=8, num_heads=2,
                        d_ff=16, num_layers=1, max_len=4)
    safe_load_model(model2, save_path, verify_hashes=False)


# ============================================
# TESTS DE DICT SEGURO
# ============================================

def test_save_load_dict(tmp_path):
    data = {
        'learning_rate': 0.01,
        'epochs': 100,
        'weights': np.random.randn(10, 10),
        'bias': np.random.randn(10),
    }
    
    save_path = str(tmp_path / "dict")
    safe_save_dict(data, save_path)
    
    loaded = safe_load_dict(save_path)
    
    assert loaded['learning_rate'] == 0.01
    assert loaded['epochs'] == 100
    assert np.array_equal(loaded['weights'], data['weights'])
    assert np.array_equal(loaded['bias'], data['bias'])


def test_load_dict_integrity(tmp_path):
    data = {'weights': np.random.randn(5, 5)}
    save_path = str(tmp_path / "dict")
    safe_save_dict(data, save_path)
    
    # Corromper
    arrays_dir = os.path.join(save_path, "arrays")
    target = os.path.join(arrays_dir, "weights.npy")
    np.save(target, np.zeros((5, 5)))
    
    with pytest.raises(ValueError, match="Hash no coincide"):
        safe_load_dict(save_path)


# ============================================
# TESTS DE SEGURIDAD
# ============================================

def test_no_pickle_in_files(model, tmp_path):
    """Los archivos generados NO deben contener código ejecutable"""
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    # Leer metadata.json (es texto plano)
    with open(os.path.join(save_path, "metadata.json"), 'r') as f:
        content = f.read()
    
    # No debe contener código sospechoso
    assert "eval" not in content
    assert "__import__" not in content
    assert "exec" not in content


def test_tensors_are_plain_npy(model, tmp_path):
    """Los tensores son archivos .npy estándar"""
    save_path = str(tmp_path / "model")
    safe_save_model(model, save_path)
    
    tensors_dir = os.path.join(save_path, "tensors")
    for f in os.listdir(tensors_dir):
        filepath = os.path.join(tensors_dir, f)
        # Debe poder cargarse con np.load sin pickle
        data = np.load(filepath, allow_pickle=False)
        assert isinstance(data, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
