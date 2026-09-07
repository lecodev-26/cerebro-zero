import numpy as np
from memoria import CerebroConMemoria
from datos import GestorDatos

print("🧠 CEREBRO CON MEMORIA")
print("="*40)

# Generar datos secuenciales (seno + ruido)
X, y = GestorDatos.generar_seno(200, ruido=0.05)

# Crear cerebro con memoria de tamaño 5
cerebro = CerebroConMemoria(entrada_size=1, memoria_size=5)

print("\n📊 Datos generados")
print(f"   Entradas: {X.shape}")
print(f"   Salidas: {y.shape}")

# Entrenar
error = cerebro.entrenar(X, y, epochs=500)

# Probar predicción
print("\n🔮 Predicción con memoria:")
for i in [10, 20, 30]:
    pred = cerebro.predecir(X[i])
    print(f"   Entrada {i}: {X[i][0]:.3f} → Predicción: {pred[0][0]:.3f} (Real: {y[i][0]:.3f})")

cerebro.guardar("cerebro_con_memoria.npy")
print("\n✅ Cerebro con memoria listo!")
