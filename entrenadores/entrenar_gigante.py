import sys
sys.path.append('..')
import numpy as np
from cerebros.cerebro_gigante import CerebroGigante

class EntrenadorGigante:
    def __init__(self, cerebro, tasa=0.0001):  # Tasa más baja
        self.cerebro = cerebro
        self.tasa = tasa
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        salida = self.cerebro.forward(X)
        error = y - salida
        grad = error
        
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            self.cerebro.pesos[i] += self.tasa * np.dot(self.cerebro.activaciones[i].T, grad)
            self.cerebro.sesgos[i] += self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.relu_derivada(self.cerebro.lineales[i-1])
                # Recortar gradientes para evitar explosión
                grad = np.clip(grad, -1, 1)
        
        return self.mse(y, salida)
    
    def entrenar(self, X, y, epochs=100):
        print(f"🏋️ ENTRENANDO CEREBRO GIGANTE...")
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        return error

def ejecutar_gigante():
    print("🧠 CEREBRO GIGANTE (VERSIÓN ESTABLE)")
    print("="*50)
    
    # Normalizar datos de entrada
    X_raw = np.random.randn(1000, 10) * 2
    # Escalar para que no explote
    X = X_raw / 10.0
    y = np.sin(X_raw.sum(axis=1, keepdims=True)) + np.random.randn(1000, 1) * 0.1
    # Escalar y
    y = y / 2.0
    
    cerebro = CerebroGigante([10, 64, 64, 1])  # Más pequeño para estabilidad
    entrenador = EntrenadorGigante(cerebro, tasa=0.0005)
    
    print(f"\n📊 Datos: {len(X)} ejemplos con 10 características (normalizados)")
    print("🏋️ Entrenando...")
    
    entrenador.entrenar(X, y, epochs=200)
    
    print("\n✅ ENTRENADO!")
    print("\n📈 Predicciones de muestra:")
    for i in range(5):
        pred = cerebro.predecir(X[i:i+1])
        real = y[i]
        print(f"   Ejemplo {i+1}: predicción = {pred[0][0]:.3f} (real: {real[0]:.3f})")
    
    cerebro.guardar("../modelos_guardados/cerebro_gigante.pkl")
    print("\n💾 Cerebro gigante guardado!")

if __name__ == "__main__":
    ejecutar_gigante()
