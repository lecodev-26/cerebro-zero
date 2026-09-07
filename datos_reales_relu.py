import numpy as np
from red_neuronal_relu import CerebroReLU

class EntrenadorReLU:
    def __init__(self, cerebro, tasa=0.001):  # Tasa más baja
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
        
        return self.mse(y, salida)
    
    def entrenar(self, X, y, epochs=5000):
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: error = {error:.6f}")
        return error

def ejecutar_datos_reales_relu():
    print("🧠 ENTRENANDO SENO CON RELU + SALIDA LINEAL")
    print("="*50)
    
    # Generar datos
    X_raw = np.random.rand(1000, 1) * 10
    y_raw = np.sin(X_raw) + np.random.randn(1000, 1) * 0.05
    
    # Normalizar X (0 a 1)
    X = X_raw / 10.0
    
    # Normalizar y (-1 a 1 ya está, pero limitamos para evitar explosión)
    y = np.clip(y_raw, -1, 1)  # Recortar valores extremos
    
    escala_x = 10.0
    
    # Cerebro más pequeño para evitar overfitting
    cerebro = CerebroReLU([1, 16, 1])  # Menos neuronas
    cerebro.resumen()
    entrenador = EntrenadorReLU(cerebro, tasa=0.001)  # Tasa muy baja
    
    print(f"\n📊 Datos: {len(X)} ejemplos")
    print("\n🏋️ ENTRENANDO...")
    entrenador.entrenar(X, y, epochs=5000)
    
    print("\n✅ ENTRENADO!")
    print("\n📈 Predicciones:")
    for x_orig in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        x_norm = x_orig / escala_x
        pred = cerebro.predecir([x_norm])
        real = np.sin(x_orig)
        print(f"  x={x_orig:.1f} → predicción: {pred[0][0]:.3f} (real: {real:.3f})")
    
    cerebro.guardar("cerebro_seno_relu.pkl")
    print("\n💾 Guardado en cerebro_seno_relu.pkl")

if __name__ == "__main__":
    ejecutar_datos_reales_relu()
