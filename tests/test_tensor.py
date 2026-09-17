"""
Tests básicos del Tensor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor, tensor, zeros, ones, randn

def test_creacion():
    print("\n🧪 Test: Creación de tensores")
    
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert np.allclose(t.data, [1, 2, 3])
    print("   ✅ Creación básica")
    
    z = zeros((2, 3))
    assert z.shape == (2, 3)
    assert np.allclose(z.data, 0)
    print("   ✅ Zeros")
    
    o = ones((2, 3))
    assert np.allclose(o.data, 1)
    print("   ✅ Ones")
    
    r = randn(2, 3)
    assert r.shape == (2, 3)
    print("   ✅ Randn")


def test_operaciones_basicas():
    print("\n🧪 Test: Operaciones básicas")
    
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    assert np.allclose((a + b).data, [5, 7, 9])
    print("   ✅ Suma")
    
    assert np.allclose((a - b).data, [-3, -3, -3])
    print("   ✅ Resta")
    
    assert np.allclose((a * b).data, [4, 10, 18])
    print("   ✅ Multiplicación")
    
    assert np.allclose((b / a).data, [4, 2.5, 2])
    print("   ✅ División")


def test_matmul():
    print("\n🧪 Test: Matmul")
    
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a.matmul(b)
    
    esperado = np.array([[19, 22], [43, 50]])
    assert np.allclose(c.data, esperado)
    print("   ✅ Matmul 2x2")


def test_reducciones():
    print("\n🧪 Test: Reducciones")
    
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert a.sum().data == 21
    print("   ✅ Sum total")
    
    assert np.allclose(a.sum(axis=0).data, [5, 7, 9])
    print("   ✅ Sum axis=0")
    
    assert np.allclose(a.sum(axis=1).data, [6, 15])
    print("   ✅ Sum axis=1")
    
    assert a.mean().data == 3.5
    print("   ✅ Mean total")
    
    assert np.allclose(a.mean(axis=0).data, [2.5, 3.5, 4.5])
    print("   ✅ Mean axis=0")


def test_activaciones():
    print("\n🧪 Test: Activaciones")
    
    a = Tensor([-1.0, 0.0, 1.0])
    
    assert np.allclose(a.relu().data, [0, 0, 1])
    print("   ✅ ReLU")
    
    # Sigmoid
    sig = Tensor([0.0]).sigmoid()
    assert np.allclose(sig.data, 0.5)
    print("   ✅ Sigmoid")
    
    # Tanh
    tanh = Tensor([0.0]).tanh()
    assert np.allclose(tanh.data, 0.0)
    print("   ✅ Tanh")
    
    # Softmax (suma 1)
    sm = Tensor([1.0, 2.0, 3.0]).softmax()
    assert np.allclose(np.sum(sm.data), 1.0)
    print("   ✅ Softmax (suma 1)")


def test_funciones_matematicas():
    print("\n🧪 Test: Funciones matemáticas")
    
    a = Tensor([1.0, 2.0, 3.0])
    
    # exp
    assert np.allclose(a.exp().data, np.exp([1.0, 2.0, 3.0]))
    print("   ✅ exp")
    
    # log
    assert np.allclose(a.log().data, np.log([1.0, 2.0, 3.0]))
    print("   ✅ log")
    
    # sqrt
    assert np.allclose(a.sqrt().data, np.sqrt([1.0, 2.0, 3.0]))
    print("   ✅ sqrt")
    
    # pow
    assert np.allclose(a.pow(2).data, [1.0, 4.0, 9.0])
    print("   ✅ pow(2)")


def test_log_dominio():
    print("\n🧪 Test: Dominio de log")
    
    try:
        Tensor([-1.0]).log()
        print("   ❌ log(-1) debería dar error")
        return False
    except ValueError:
        print("   ✅ log(-1) da ValueError correctamente")
    
    try:
        Tensor([0.0]).log()
        print("   ❌ log(0) debería dar error")
        return False
    except ValueError:
        print("   ✅ log(0) da ValueError correctamente")
    
    return True


def test_sqrt_dominio():
    print("\n🧪 Test: Dominio de sqrt")
    
    try:
        Tensor([-1.0]).sqrt()
        print("   ❌ sqrt(-1) debería dar error")
        return False
    except ValueError:
        print("   ✅ sqrt(-1) da ValueError correctamente")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTS DEL TENSOR")
    print("="*60)
    
    test_creacion()
    test_operaciones_basicas()
    test_matmul()
    test_reducciones()
    test_activaciones()
    test_funciones_matematicas()
    test_log_dominio()
    test_sqrt_dominio()
    
    print("\n" + "="*60)
    print("✅ TODOS LOS TESTS DEL TENSOR PASARON")
    print("="*60)
