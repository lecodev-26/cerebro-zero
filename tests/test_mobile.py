"""
Tests de optimizaciones móviles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile
from mobile.optimizer import MobileOptimizer
from mobile.metrics import ResourceMonitor
from mobile.quantizer import Quantizer
from core.tensor import Tensor, get_dtype, set_dtype


# ============================================
# FIXTURES
# ============================================

@pytest.fixture(autouse=True)
def restore_dtype():
    """Restaurar dtype original después de cada test"""
    original = get_dtype()
    yield
    set_dtype(original)


@pytest.fixture
def temp_npz_path(tmp_path):
    """Devuelve un path temporal para guardar npz"""
    return str(tmp_path / "test_quant.npz")


# ============================================
# TESTS DEL OPTIMIZER
# ============================================

def test_optimizer_creation():
    opt = MobileOptimizer()
    assert not opt.float32_enabled


def test_optimizer_enable_float32():
    opt = MobileOptimizer()
    opt.enable_float32()
    assert opt.float32_enabled
    assert get_dtype() == np.float32


def test_optimizer_disable_float32():
    opt = MobileOptimizer()
    opt.enable_float32()
    opt.disable_float32()
    assert not opt.float32_enabled


def test_optimizer_cache():
    opt = MobileOptimizer()
    opt.cache_result("test", 42)
    assert opt.get_cached("test") == 42


def test_optimizer_cache_miss():
    opt = MobileOptimizer()
    assert opt.get_cached("nonexistent") is None


def test_optimizer_clear_cache():
    opt = MobileOptimizer()
    opt.cache_result("a", 1)
    opt.clear_cache()
    assert opt.get_cached("a") is None


def test_optimizer_float32_tensor():
    opt = MobileOptimizer()
    opt.enable_float32()
    t = Tensor([1.0, 2.0, 3.0])
    assert t.data.dtype == np.float32


# ============================================
# TESTS DEL RESOURCE MONITOR
# ============================================

def test_monitor_creation():
    monitor = ResourceMonitor()
    assert len(monitor.measurements) == 0


def test_monitor_measure():
    import time
    monitor = ResourceMonitor()
    with monitor.measure("test"):
        time.sleep(0.01)
    assert len(monitor.measurements) == 1
    assert monitor.measurements[0]['time_ms'] >= 10


def test_monitor_multiple_measurements():
    import time
    monitor = ResourceMonitor()
    for _ in range(3):
        with monitor.measure("test"):
            time.sleep(0.001)
    assert len(monitor.measurements) == 3


def test_monitor_stats():
    import time
    monitor = ResourceMonitor()
    for _ in range(3):
        with monitor.measure("test"):
            time.sleep(0.001)
    stats = monitor.get_stats()
    assert stats['count'] == 3
    assert stats['total_ms'] > 0


def test_monitor_clear():
    import time
    monitor = ResourceMonitor()
    with monitor.measure("test"):
        time.sleep(0.001)
    monitor.clear()
    assert len(monitor.measurements) == 0


# ============================================
# TESTS DEL QUANTIZER
# ============================================

def test_quantizer_creation():
    q = Quantizer()
    assert q is not None


def test_quantize_array():
    q = Quantizer()
    arr = np.random.randn(10, 10).astype(np.float32)
    quantized, scale, zero_point = q.quantize_array(arr, bits=8)
    assert quantized.dtype == np.uint8
    assert quantized.shape == arr.shape


def test_quantize_dequantize():
    q = Quantizer()
    arr = np.random.randn(100, 100).astype(np.float32) * 10
    quantized, scale, zero_point = q.quantize_array(arr, bits=8)
    restored = q.dequantize_array(quantized, scale, zero_point)
    
    error = np.mean(np.abs(arr - restored))
    arr_range = np.max(arr) - np.min(arr)
    assert error < arr_range * 0.01, f"Error demasiado grande: {error}"


def test_quantize_compression():
    q = Quantizer()
    arr = np.random.randn(100, 100).astype(np.float32)
    quantized, _, _ = q.quantize_array(arr, bits=8)
    assert quantized.nbytes == arr.nbytes // 4


def test_quantize_model(temp_npz_path):
    """Usa tmp_path de pytest para evitar problemas de permisos"""
    from models.transformer import Transformer
    q = Quantizer()
    model = Transformer(vocab_size=20, d_model=8, num_heads=2,
                        d_ff=16, num_layers=1, max_len=8)
    
    result = q.quantize_model(model, temp_npz_path)
    assert result['num_params'] > 0
    assert result['compression_ratio'] > 1
    assert os.path.exists(temp_npz_path)


def test_quantize_constant_array():
    """Array constante (min == max)"""
    q = Quantizer()
    arr = np.ones((10, 10), dtype=np.float32)
    quantized, scale, zero_point = q.quantize_array(arr, bits=8)
    restored = q.dequantize_array(quantized, scale, zero_point)
    assert not np.any(np.isnan(restored))


# ============================================
# TESTS DE INTEGRACIÓN
# ============================================

def test_float32_faster_than_float64():
    """Verificar que float32 es al menos tan rápido como float64"""
    import time
    
    set_dtype(np.float64)
    a = Tensor(np.random.randn(500, 500))
    b = Tensor(np.random.randn(500, 500))
    start = time.time()
    for _ in range(5):
        c = a.matmul(b)
    time_f64 = time.time() - start
    
    set_dtype(np.float32)
    a = Tensor(np.random.randn(500, 500))
    b = Tensor(np.random.randn(500, 500))
    start = time.time()
    for _ in range(5):
        c = a.matmul(b)
    time_f32 = time.time() - start
    
    assert time_f32 <= time_f64 * 1.2, (
        f"float32 no es más rápido: f32={time_f32:.3f}, f64={time_f64:.3f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
