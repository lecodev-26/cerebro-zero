import numpy as np
from red_neuronal import Cerebro

class Entrenador:
    def __init__(self, cerebro, tasa=0.5):
        self.cerebro = cerebro
        self.tasa = tasa
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        # Forward pass
        salida = self.cerebro.forward(X)
        
        # Calcular error
        error = y - salida
        
        # Calcular gradiente de salida (sigmoid derivative)
        grad = error * salida * (1 - salida)
        
        # Backpropagation
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            # Actualizar pesos y sesgos de la capa actual
            self.cerebro.pesos[i] += self.tasa * np.dot(self.cerebro.activaciones[i].T, grad)
            self.cerebro.sesgos[i] += self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            # Calcular gradiente para la capa anterior (si existe)
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.activaciones[i] * (1 - self.cerebro.activaciones[i])
        
        return self.mse(y, salida)
    
    def entrenar(self, X, y, epochs=10000, verbose=True):
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}: error = {error:.6f}")
        return error

# DATOS DE EJEMPLO: XOR
if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    cerebro = Cerebro([2, 4, 1])
    entrenador = Entrenador(cerebro, tasa=0.8)
    
    print("🧠 Entrenando cerebro para XOR...")
    print("Esto tardará unos segundos cuando numpy esté listo")
    
    for epoch in range(10000):
        error = entrenador.entrenar_epoch(X, y)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: error = {error:.6f}")
    
    print("\n✅ RESULTADOS FINALES:")
    for inputs in X:
        pred = cerebro.predecir(inputs)
        print(f"{inputs} -> {pred[0][0]:.4f}")
    
    cerebro.guardar("xor_cerebro.npy")
