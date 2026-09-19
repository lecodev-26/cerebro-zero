import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.mlp import MLP
from core.optimizer import Adam

print("🧠 CEREBRO GIGANTE MEJORADO")
print("="*40)
print("🔄 2 capas ocultas: 256 → 128 neuronas")
print("📊 Parámetros totales: ~33,000")
print("="*40)

# Crear cerebro más grande
model = MLP([2, 256, 128, 1])
optimizer = Adam(model.layers, lr=0.01)

# Datos XOR con más ejemplos
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
X_train = np.repeat(X, 200, axis=0)
y_train = np.repeat(y, 200, axis=0)

print("\n🏋️ ENTRENANDO...")
for epoch in range(10000):
    loss = model.train_step(X_train, y_train, optimizer)
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: loss = {loss:.6f}")

print("\n✅ RESULTADOS:")
for x in X:
    pred = model.predict(np.array([x]))[0][0]
    print(f"   {x} → {pred:.4f}")

print("\n🧠 CEREBRO GIGANTE MEJORADO FUNCIONANDO!")
