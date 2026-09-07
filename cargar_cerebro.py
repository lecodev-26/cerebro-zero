from red_neuronal import Cerebro

# Crear cerebro con la misma estructura
cerebro = Cerebro([2, 8, 1])

# Cargar el cerebro guardado
cerebro.cargar("xor_cerebro_mejorado.pkl")

# Probar predicciones
print("\n🧠 PRUEBA DEL CEREBRO CARGADO:")
print("="*30)
print(f"[0,0] -> {cerebro.predecir([0,0])[0][0]:.4f}")
print(f"[0,1] -> {cerebro.predecir([0,1])[0][0]:.4f}")
print(f"[1,0] -> {cerebro.predecir([1,0])[0][0]:.4f}")
print(f"[1,1] -> {cerebro.predecir([1,1])[0][0]:.4f}")
