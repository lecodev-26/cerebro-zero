"""
Tests del cargador de configuración
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import yaml
from config.loader import Config, get_config, reset_config


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def sample_config_data():
    return {
        'model': {
            'type': 'transformer',
            'd_model': 32,
            'num_layers': 2,
        },
        'training': {
            'learning_rate': 0.01,
            'batch_size': 8,
        },
        'memory': {
            'short_term': {'max_size': 100},
        },
    }


@pytest.fixture
def config_file(sample_config_data, tmp_path):
    path = tmp_path / "test_config.yaml"
    with open(path, 'w') as f:
        yaml.dump(sample_config_data, f)
    return str(path)


@pytest.fixture(autouse=True)
def reset():
    """Resetear config global antes de cada test"""
    reset_config()
    yield
    reset_config()


# ============================================
# TESTS BÁSICOS
# ============================================

def test_config_creation():
    config = Config({'a': 1})
    assert config.data == {'a': 1}


def test_config_empty():
    config = Config()
    assert config.data == {}


def test_load_from_file(config_file):
    config = Config.load(config_file)
    assert config.get('model.type') == 'transformer'
    assert config.get('training.learning_rate') == 0.01


def test_load_nonexistent():
    config = Config.load('nonexistent_file.yaml')
    assert config.data == {}


# ============================================
# TESTS DE GET
# ============================================

def test_get_simple(config_file):
    config = Config.load(config_file)
    assert config.get('model.type') == 'transformer'


def test_get_nested(config_file):
    config = Config.load(config_file)
    assert config.get('model.d_model') == 32
    assert config.get('model.num_layers') == 2


def test_get_deep_nested(config_file):
    config = Config.load(config_file)
    assert config.get('memory.short_term.max_size') == 100


def test_get_with_default(config_file):
    config = Config.load(config_file)
    assert config.get('nonexistent', 'default') == 'default'
    assert config.get('model.nonexistent', 42) == 42


def test_get_returns_none_for_missing(config_file):
    config = Config.load(config_file)
    assert config.get('nonexistent') is None


# ============================================
# TESTS DE SET
# ============================================

def test_set_simple():
    config = Config({})
    config.set('a', 1)
    assert config.get('a') == 1


def test_set_nested():
    config = Config({})
    config.set('a.b.c', 42)
    assert config.get('a.b.c') == 42


def test_set_overwrites():
    config = Config({'a': 1})
    config.set('a', 2)
    assert config.get('a') == 2


# ============================================
# TESTS DE SECTION
# ============================================

def test_section(config_file):
    config = Config.load(config_file)
    model = config.section('model')
    assert model['type'] == 'transformer'
    assert model['d_model'] == 32


def test_section_missing(config_file):
    config = Config.load(config_file)
    section = config.section('nonexistent')
    assert section == {}


# ============================================
# TESTS DE TO_DICT
# ============================================

def test_to_dict(config_file):
    config = Config.load(config_file)
    d = config.to_dict()
    assert d['model']['type'] == 'transformer'


def test_to_dict_is_copy(config_file):
    config = Config.load(config_file)
    d1 = config.to_dict()
    d1['model']['type'] = 'CHANGED'
    d2 = config.to_dict()
    assert d2['model']['type'] == 'transformer'  # No cambió


# ============================================
# TESTS DEL DEFAULT.YAML
# ============================================

def test_default_config_exists():
    """El archivo default.yaml debe existir y ser válido"""
    assert os.path.exists('config/default.yaml')
    
    config = Config.load('config/default.yaml')
    assert config.get('model.type') == 'transformer'
    assert config.get('training.learning_rate') is not None
    assert config.get('memory.short_term.max_size') is not None
    assert config.get('security.sandbox.strict') is not None


def test_default_config_has_all_sections():
    config = Config.load('config/default.yaml')
    sections = ['model', 'training', 'memory', 'datasets',
                'security', 'continuous_learning', 'evaluation',
                'paths', 'logging']
    for section in sections:
        assert config.get(section) is not None, f"Falta sección '{section}'"


def test_default_config_types():
    config = Config.load('config/default.yaml')
    assert isinstance(config.get('model.d_model'), int)
    assert isinstance(config.get('training.learning_rate'), float)
    assert isinstance(config.get('security.sandbox.strict'), bool)


# ============================================
# TESTS DE CONFIG GLOBAL
# ============================================

def test_get_config_singleton():
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2


def test_reset_config():
    c1 = get_config()
    reset_config()
    c2 = get_config()
    assert c1 is not c2


# ============================================
# TESTS DE REPR
# ============================================

def test_repr(config_file):
    config = Config.load(config_file)
    r = repr(config)
    assert 'Config' in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
