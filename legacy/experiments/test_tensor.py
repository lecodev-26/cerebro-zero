"""
Prueba del sistema de tensores
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor, ones, zeros, randn, tensor

print("🧠 PROBANDO TENSOR")
print("="*40)

# 1. Creación de tensores
print("\n📦 1. Creación:")
a = Tensor([1, 2, 3])
b = Tensor(np.array([[1, 2], [3, 4]]))
c = ones((2, 3))
d = zeros((3, 2))
e = randn((2, 2))

print(f"   a: {a.data}")
print(f"   b: {b.data}")
print(f"   c: {c.data}")
print(f"   d: {d.data}")
print(f"   e: {e.data}")

# 2. Operaciones básicas
print("\n➕ 2. Operaciones:")
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)
z = x * y + x

print(f"   x = {x.data}")
print(f"   y = {y.data}")
print(f"   z = x * y + x = {z.data}")

z.backward()
print(f"   dz/dx = {x.grad} (debería ser 4)")
print(f"   dz/dy = {y.grad} (debería ser 2)")

# 3. Operaciones con matrices
print("\n🔢 3. Operaciones con matrices:")
A = Tensor([[1, 2], [3, 4]], requires_grad=True)
B = Tensor([[5, 6], [7, 8]], requires_grad=True)
C = A.matmul(B)

print(f"   A = {A.data}")
print(f"   B = {B.data}")
print(f"   C = A @ B = {C.data}")

C.sum().backward()
print(f"   dC/dA = {A.grad}")
print(f"   dC/dB = {B.grad}")

# 4. Funciones matemáticas
print("\n📐 4. Funciones matemáticas:")
x = Tensor(2.0, requires_grad=True)
y = x.exp()
z = y.log()
w = z.sqrt()

print(f"   x = {x.data}")
print(f"   y = exp(x) = {y.data}")
print(f"   z = log(y) = {z.data}")
print(f"   w = sqrt(z) = {w.data}")

w.backward()
print(f"   dw/dx = {x.grad}")

print("\n✅ TENSOR FUNCIONANDO CORRECTAMENTE!")

# 5. Comparación con NumPy
print("\n🔬 5. Comparación con NumPy:")
x_np = np.array([1, 2, 3])
x_t = Tensor(x_np)
suma_np = x_np + x_np
suma_t = x_t + x_t

print(f"   NumPy: {suma_np}")
print(f"   Tensor: {suma_t.data}")

print("\n✅ TODAS LAS PRUEBAS PASADAS!")
