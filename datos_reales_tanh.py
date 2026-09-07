import numpy as np
from red_neuronal_tanh import CerebroTanh

class EntrenadorTanh:
    def __init__(self, cerebro, tasa=0.3):
        self.cerebro = cerebro
        self.tasa = tasa
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        salida = self.cerebro.forward(X, entrenando=True)
        error = y - salida
        grad = error * self.cerebro.tanh_derivada(self.cerebro.lineales[-1])
        
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            self.cerebro.pesos[i] += self.tasa * np.dot(self.cerebro.activaciones[i].T, grad)
            self.cerebro.sesgos[i] += self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.activaciones[i] * (1 - self.cerebro.activaciones[i])
        
        return self.mse(y, salida)
    
    def entrenar(self, X, y, epochs=10000):
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: error = {error:.6f}")
        return error

def ejecutar_datos_reales_tanh():
    print("🧠 ENTRENANDO SENO CON TANH (valores -1 a 1)")
    print("="*50)
    
    # Generar datos
    X_raw = np.random.rand(1000, 1) * 10
    y = np.sin(X_raw) + np.random.randn(1000, 1) * 0.05
    
    # NORMALIZAR X (escalar entre 0 y 1)
    X = X_raw / 10.0  # Ahora X está entre 0 y 1
    
    # Guardar el factor de escala para predicciones
    escala = 10.0
    
    cerebro = CerebroTanh([1, 20, 1])
    cerebro.resumen()
    entrenador = EntrenadorTanh(cerebro, tasa=0.5)
    
    print(f"\n📊 Datos: {len(X)} ejemplos (X normalizado entre 0 y 1)")
    print("\n🏋️ ENTRENANDO...")
    entrenador.entrenar(X, y, epochs=10000)
    
    print("\n✅ ENTRENADO!")
    print("\n📈 Predicciones (X original):")
    for x_orig in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        x_norm = x_orig / escala  # Normalizar para la predicción
        pred = cerebro.predecir([x_norm])
        real = np.sin(x_orig)
        print(f"  x={x_orig:.1f} → predicción: {pred[0][0]:.3f} (real: {real:.3f})")
    
    # Guardar también la escala
    cerebro.guardar("cerebro_seno_tanh.pkl")
    with open("escala_seno.txt", "w") as f:
        f.write(str(escala))
    print("\n💾 Guardado en cerebro_seno_tanh.pkl y escala_seno.txt")

if __name__ == "__main__":
    ejecutar_datos_reales_tanh()
