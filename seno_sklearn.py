import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

print("🧠 ENTRENANDO SENO CON SCIKIT-LEARN")
print("="*50)

# Generar datos
X = np.random.rand(1000, 1) * 10
y = np.sin(X).ravel() + np.random.randn(1000) * 0.05

# Crear red neuronal
modelo = MLPRegressor(
    hidden_layer_sizes=(20, 10),
    activation='relu',
    solver='adam',
    max_iter=5000,
    random_state=42
)

print("🏋️ ENTRENANDO...")
modelo.fit(X, y)

print("\n✅ ENTRENADO!")
print(f"   Error final: {modelo.loss_:.6f}")

print("\n📈 Predicciones:")
for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    pred = modelo.predict([[x]])[0]
    real = np.sin(x)
    print(f"  x={x:.1f} → predicción: {pred:.3f} (real: {real:.3f})")

# Guardar modelo
with open("seno_sklearn.pkl", "wb") as f:
    pickle.dump(modelo, f)
print("\n💾 Guardado en seno_sklearn.pkl")
