import numpy as np
from red_neuronal import Cerebro
from entrenar import Entrenador

print("🧠 CREANDO CEREBRO CON DATOS REALES")
print("="*40)

# Generar datos: función seno con ruido
X = np.random.rand(500, 1) * 10  # 500 números entre 0 y 10
y = np.sin(X) + np.random.randn(500, 1) * 0.1  # Seno con ruido

# Crear cerebro: 1 entrada, 10 ocultas, 1 salida
cerebro = Cerebro([1, 10, 1])
entrenador = Entrenador(cerebro, tasa=0.3)

print(f"📊 Datos: {len(X)} ejemplos")
print(f"🧠 Estructura: {cerebro.capas}")
print("\n🏋️ ENTRENANDO...")

for epoch in range(5000):
    error = entrenador.entrenar_epoch(X, y)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: error = {error:.6f}")

print("\n✅ ENTRENADO!")
print("\n📈 Predicciones:")
for x in [0, 2, 4, 6, 8, 10]:
    pred = cerebro.predecir([x])
    real = np.sin(x)
    print(f"  x={x:.1f} → predicción: {pred[0][0]:.3f} (real: {real:.3f})")

cerebro.guardar("cerebro_seno.npy")
print("\n💾 Guardado en cerebro_seno.npy")
