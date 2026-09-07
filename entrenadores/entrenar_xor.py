import sys
sys.path.append('..')
import numpy as np
from cerebros.red_neuronal import Cerebro

class Entrenador:
    def __init__(self, cerebro, tasa=0.5):
        self.cerebro = cerebro
        self.tasa = tasa
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        salida = self.cerebro.forward(X)
        error = y - salida
        grad = error * salida * (1 - salida)
        
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            self.cerebro.pesos[i] += self.tasa * np.dot(self.cerebro.activaciones[i].T, grad)
            self.cerebro.sesgos[i] += self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.activaciones[i] * (1 - self.cerebro.activaciones[i])
        
        return self.mse(y, salida)

def entrenar_xor():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    cerebro = Cerebro([2, 8, 1])
    cerebro.resumen()
    entrenador = Entrenador(cerebro, tasa=0.8)
    
    print("\n🧠 ENTRENANDO XOR...")
    for epoch in range(20000):
        error = entrenador.entrenar_epoch(X, y)
        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: error = {error:.6f}")
    
    print("\n✅ RESULTADOS FINALES:")
    for inputs in X:
        pred = cerebro.predecir(inputs)
        print(f"{inputs} -> {pred[0][0]:.4f}")
    
    cerebro.guardar("../modelos_guardados/xor_cerebro_mejorado.pkl")
    print("\n💾 Cerebro guardado correctamente!")

if __name__ == "__main__":
    entrenar_xor()
