"""
Prueba de la Fase 2: XOR con BCE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor
from core.layers import Linear, Sequential
from core.activations import ReLU, Sigmoid
from core.losses import BCELoss

print("🧠 PROBANDO XOR CON BCE")
print("="*40)

# Datos XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
Y = np.array([[0],[1],[1],[0]], dtype=np.float64)

x_t = Tensor(X)
y_t = Tensor(Y)

# Red más grande
model = Sequential(
    Linear(2, 16),
    ReLU(),
    Linear(16, 16),
    ReLU(),
    Linear(16, 1),
    Sigmoid()
)

print(f"Modelo:\n{model}")

loss_fn = BCELoss()  # <-- Usamos BCE
lr = 0.8

print("\n🏋️ ENTRENANDO XOR CON BCE...")
for epoch in range(20000):
    # Forward
    pred = model(x_t)
    loss = loss_fn(pred, y_t)
    
    # Backward
    loss.backward()
    
    # Update
    for param in model.parameters():
        if param.grad is not None:
            param.data -= lr * param.grad
    
    # Reset gradients
    model.zero_grad()
    
    if epoch % 2000 == 0:
        print(f"   Epoch {epoch}: loss = {loss.data:.6f}")

print("\n✅ RESULTADOS FINALES:")
for i, x in enumerate(X):
    pred = model(Tensor(x)).data
    valor = pred[0, 0]
    print(f"   {x} → {valor:.4f}")

print("\n✅ FASE 2 COMPLETADA CORRECTAMENTE!")
