import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.mlp import MLP
from core.optimizer import Adam
from memory.memory import MemoriaPersistente

print("🧠 CEREBRO CON MEMORIA PERSISTENTE")
print("="*40)

# Crear cerebro
model = MLP([2, 8, 1])
optimizer = Adam(model.layers, lr=0.1)

# Crear memoria
memoria = MemoriaPersistente(archivo="cerebro_memoria.json", max_size=100)

# Datos iniciales (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

print("\n🏋️ ENTRENANDO CEREBRO CON XOR...")
for epoch in range(5000):
    loss = model.train_step(X, y, optimizer)
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: loss = {loss:.6f}")

print("\n✅ CEREBRO ENTRENADO!")
for x in X:
    pred = model.predict(np.array([x]))[0][0]
    print(f"   {x} → {pred:.4f}")

# Guardar experiencia
print("\n📝 GUARDANDO EXPERIENCIA EN MEMORIA...")
for x, y_true in zip(X, y):
    pred = model.predict(np.array([x]))[0][0]
    experiencia = np.array([x[0], x[1], pred])
    memoria.aprender(np.array([x[0], x[1]]), np.array([pred]))
    print(f"   {x} → {pred:.4f} (guardado)")

# Recuperar experiencias similares
print("\n🔮 RECUPERANDO EXPERIENCIAS SIMILARES...")
prueba = [0.5, 0.5]
similares = memoria.recordar(prueba, k=3)
print(f"   Experiencias similares a {prueba}:")
for s in similares:
    print(f"      {s['entrada']} → {s['salida']}")

print("\n🧠 ¡CEREBRO CON MEMORIA FUNCIONANDO!")
