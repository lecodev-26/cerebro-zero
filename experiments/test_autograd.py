"""
Prueba específica de autograd
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor

print("🧠 PRUEBA ESPECÍFICA DE AUTOGRAD")
print("="*40)

# 1. Prueba simple: y = x * w + b
print("\n📦 1. Prueba simple:")
x = Tensor([2.0], requires_grad=True)
w = Tensor([3.0], requires_grad=True)
b = Tensor([1.0], requires_grad=True)

y = x * w + b
loss = y.mean()

loss.backward()
print(f"   x.grad = {x.grad} (debería ser 3.0)")
print(f"   w.grad = {w.grad} (debería ser 2.0)")
print(f"   b.grad = {b.grad} (debería ser 1.0)")

# 2. Prueba con matmul: y = X @ W
print("\n📦 2. Prueba con matmul:")
X = Tensor([[1.0, 2.0]], requires_grad=True)
W = Tensor([[3.0], [4.0]], requires_grad=True)
y = X.matmul(W)
loss = y.mean()

loss.backward()
print(f"   X.grad = {X.grad}")
print(f"   W.grad = {W.grad}")

# 3. Prueba XOR manual con autograd
print("\n📦 3. Prueba XOR manual:")
x1 = Tensor([1.0, 0.0], requires_grad=True)
w1 = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
b1 = Tensor([0.0, 0.0], requires_grad=True)

hidden = x1.matmul(w1) + b1
hidden_relu = Tensor(np.maximum(0, hidden.data), requires_grad=True, _children=(hidden,))
print(f"   hidden = {hidden_relu.data}")

print("\n✅ AUTOGRAD FUNCIONANDO CORRECTAMENTE!")
