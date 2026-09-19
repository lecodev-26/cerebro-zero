import sys
import os
# Añadir la ruta raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.mlp import MLP
from core.optimizer import Adam

print("🧠 PROBANDO LA NUEVA ARQUITECTURA")
print("="*40)

# Datos XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Crear modelo
model = MLP([2, 8, 1])
optimizer = Adam(model.layers, lr=0.1)

print(f"Modelo: {model}")
print("\n🏋️ ENTRENANDO...")

for epoch in range(5000):
    loss = model.train_step(X, y, optimizer)
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: loss = {loss:.6f}")

print("\n✅ RESULTADOS:")
for x in X:
    pred = model.predict(np.array([x]))[0][0]
    print(f"   {x} → {pred:.4f}")

print("\n🧠 ¡NUEVA ARQUITECTURA FUNCIONANDO!")
