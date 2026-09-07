import sys
sys.path.append('..')
import numpy as np
import pickle

print("🧠 APRENDIENDO SENO CON REGRESIÓN POLINÓMICA")
print("="*50)

X = np.random.rand(1000, 1) * 10
y = np.sin(X) + np.random.randn(1000, 1) * 0.05

X_poly = np.hstack([X**i for i in range(1, 6)])
X_poly = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])

w = np.linalg.lstsq(X_poly, y, rcond=None)[0]

print("✅ MODELO ENTRENADO!")
print(f"   Pesos: {w.flatten()}")

print("\n📈 Predicciones:")
for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    x_poly = np.array([1] + [x**i for i in range(1, 6)])
    pred = np.dot(x_poly, w)[0]
    real = np.sin(x)
    print(f"  x={x:.1f} → predicción: {pred:.3f} (real: {real:.3f})")

with open("../modelos_guardados/seno_polinomico.pkl", "wb") as f:
    pickle.dump({'pesos': w, 'grado': 5}, f)
print("\n💾 Guardado en modelos_guardados/seno_polinomico.pkl")
