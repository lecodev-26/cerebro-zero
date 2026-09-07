import numpy as np
from red_neuronal import Cerebro
from entrenar import Entrenador

print("🧪 PRUEBAS DEL CEREBRO")
print("="*40)

# 1. Crear cerebro pequeño
cerebro = Cerebro([2, 3, 1])
cerebro.resumen()

print("\n📊 Probando propagación:")
entrada = [0.5, -0.3]
salida = cerebro.predecir(entrada)
print(f"Entrada: {entrada} → Salida: {salida[0][0]:.4f}")

# 2. Probar guardado/carga
cerebro.guardar("test.npy")
cerebro2 = Cerebro([2, 3, 1])
cerebro2.cargar("test.npy")

print("\n✅ Todas las pruebas pasadas!")
