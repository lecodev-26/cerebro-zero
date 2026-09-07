import sys
sys.path.append('..')
import numpy as np
from cerebros.cerebro_gigante import CerebroGigante

class EntrenadorDios:
    def __init__(self, cerebro, tasa=0.00001):  # Tasa MUY baja
        self.cerebro = cerebro
        self.tasa = tasa
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        salida = self.cerebro.forward(X)
        error = y - salida
        grad = error
        
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            # Recortar gradientes para evitar explosión
            grad = np.clip(grad, -1, 1)
            self.cerebro.pesos[i] += self.tasa * np.dot(self.cerebro.activaciones[i].T, grad)
            self.cerebro.sesgos[i] += self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.relu_derivada(self.cerebro.lineales[i-1])
        
        return self.mse(y, salida)
    
    def entrenar(self, X, y, epochs=50):
        print(f"🏋️ ENTRENANDO CEREBRO DE DIOS...")
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if epoch % 5 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        return error

def ejecutar_dios():
    print("🧠 CEREBRO DE DIOS (1000 neuronas - VERSIÓN ESTABLE)")
    print("="*50)
    print("⚠️ Usando tasa de aprendizaje MUY baja para evitar explosión")
    print("⚠️ Esto tardará más pero funcionará")
    print("="*50)
    
    # Crear cerebro
    cerebro = CerebroGigante([10, 1000, 1])
    
    # Datos
    X = np.random.randn(1000, 10) * 2
    y = np.sin(X.sum(axis=1, keepdims=True)) + np.random.randn(1000, 1) * 0.1
    X = X / 5.0
    y = y / 2.0
    
    entrenador = EntrenadorDios(cerebro, tasa=0.00001)  # Tasa bajísima
    
    print(f"\n📊 Datos: {len(X)} ejemplos")
    print("🏋️ ENTRENANDO... (será lento pero estable)")
    
    entrenador.entrenar(X, y, epochs=100)
    
    print("\n✅ ENTRENADO!")
    print("\n📈 Predicciones:")
    for i in range(5):
        pred = cerebro.predecir(X[i:i+1])
        print(f"   Ejemplo {i+1}: {pred[0][0]:.3f} (real: {y[i][0]:.3f})")
    
    cerebro.guardar("../modelos_guardados/cerebro_dios.pkl")
    print("\n💾 Cerebro de dios guardado!")
