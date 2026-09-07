import numpy as np
from red_neuronal import Cerebro

print("🧠 Creando cerebro con 100 neuronas ocultas...")

# 5 entradas, 100 ocultas, 2 salidas
cerebro = Cerebro([5, 100, 2])

# Datos de ejemplo (patrones aleatorios)
X = np.random.randn(100, 5)
y = np.sin(X.sum(axis=1, keepdims=True))

print(f"✅ Cerebro creado:")
print(f"   - Capa 1: {cerebro.pesos[0].shape} (5→100)")
print(f"   - Capa 2: {cerebro.pesos[1].shape} (100→2)")
print(f"   - Total de parámetros: {cerebro.pesos[0].size + cerebro.pesos[1].size}")

cerebro.guardar("cerebro_grande.npy")
