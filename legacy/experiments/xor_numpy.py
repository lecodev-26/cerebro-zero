"""
XOR con NumPy puro (sin autograd)
"""

import numpy as np

class RedNumpy:
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
            lim = np.sqrt(6.0 / (capas[i] + capas[i+1]))
            w = np.random.uniform(-lim, lim, (capas[i], capas[i+1]))
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivada(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.activaciones = [X]
        self.lineales = []
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = np.dot(self.activaciones[-1], w) + b
            self.lineales.append(z)
            if i == len(self.pesos) - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def backward(self, X, y, salida):
        error = y - salida
        grad = error * self.sigmoid_derivada(salida)
        
        for i in range(len(self.pesos)-1, -1, -1):
            self.pesos[i] += self.learning_rate * np.dot(self.activaciones[i].T, grad)
            self.sesgos[i] += self.learning_rate * np.sum(grad, axis=0, keepdims=True)
            if i > 0:
                grad = np.dot(grad, self.pesos[i].T)
                grad = grad * self.relu_derivada(self.lineales[i-1])
        
        return np.mean(error ** 2)
    
    def entrenar(self, X, y, epochs=10000, lr=0.1):
        self.learning_rate = lr
        print(f"🏋️ ENTRENANDO XOR CON NUMPY PURO...")
        for epoch in range(epochs):
            salida = self.forward(X)
            error = self.backward(X, y, salida)
            if epoch % 1000 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
    
    def predecir(self, X):
        return self.forward(X)

# Datos XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
y = np.array([[0],[1],[1],[0]], dtype=np.float64)

# Red con 2 capas ocultas
red = RedNumpy([2, 16, 16, 1])
red.entrenar(X, y, epochs=10000, lr=0.5)

print("\n✅ RESULTADOS:")
for i, x in enumerate(X):
    pred = red.predecir(x.reshape(1, -1))[0, 0]
    print(f"   {x} → {pred:.4f}")
