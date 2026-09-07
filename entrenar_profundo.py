import numpy as np
from red_profunda import CerebroProfundo

class EntrenadorProfundo:
    def __init__(self, cerebro, tasa=0.3):
        self.cerebro = cerebro
        self.tasa = tasa
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        salida = self.cerebro.forward(X, entrenando=True)
        error = y - salida
        grad = error * salida * (1 - salida)
        
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.relu_derivada(self.cerebro.lineales[i-1])
            
            self.cerebro.pesos[i] += self.tasa * np.dot(self.cerebro.activaciones[i].T, grad)
            self.cerebro.sesgos[i] += self.tasa * np.sum(grad, axis=0, keepdims=True)
        
        return self.mse(y, salida)
    
    def entrenar(self, X, y, epochs=5000):
        print(f"🏋️ ENTRENANDO CEREBRO PROFUNDO...")
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if epoch % 500 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        return error

def ejecutar_profundo():
    print("🧠 CEREBRO PROFUNDO CON 5 CAPAS")
    print("="*40)
    
    cerebro = CerebroProfundo([2, 64, 32, 16, 1])
    cerebro.resumen()
    
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    X_train = np.repeat(X, 50, axis=0)
    y_train = np.repeat(y, 50, axis=0)
    
    entrenador = EntrenadorProfundo(cerebro, tasa=0.5)
    error = entrenador.entrenar(X_train, y_train, epochs=3000)
    
    print("\n✅ RESULTADOS:")
    for inputs in X:
        pred = cerebro.predecir(inputs)
        print(f"   {inputs} → {pred[0][0]:.4f}")
    
    cerebro.guardar("profundo_xor.npy")
    print("\n💾 Guardado en profundo_xor.npy")

if __name__ == "__main__":
    ejecutar_profundo()
