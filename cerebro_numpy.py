"""
Cerebro Zero - Versión NumPy (sin autograd)
"""

import numpy as np
import pickle

class CerebroNumpy:
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
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivada(self, x):
        return 1 - x**2
    
    def forward(self, X, activacion_oculta='relu', activacion_salida='sigmoid'):
        self.activaciones = [X]
        self.lineales = []
        
        activaciones = {
            'relu': (self.relu, self.relu_derivada),
            'tanh': (self.tanh, self.tanh_derivada),
            'sigmoid': (self.sigmoid, self.sigmoid_derivada)
        }
        
        self.f_oculta, self.df_oculta = activaciones[activacion_oculta]
        self.f_salida, self.df_salida = activaciones[activacion_salida]
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = np.dot(self.activaciones[-1], w) + b
            self.lineales.append(z)
            if i == len(self.pesos) - 1:
                a = self.f_salida(z)
            else:
                a = self.f_oculta(z)
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def backward(self, X, y, salida, lr=0.1):
        error = y - salida
        grad = error * self.df_salida(salida)
        
        for i in range(len(self.pesos)-1, -1, -1):
            self.pesos[i] += lr * np.dot(self.activaciones[i].T, grad)
            self.sesgos[i] += lr * np.sum(grad, axis=0, keepdims=True)
            if i > 0:
                grad = np.dot(grad, self.pesos[i].T)
                grad = grad * self.df_oculta(self.lineales[i-1])
        
        return np.mean(error ** 2)
    
    def entrenar(self, X, y, epochs=10000, lr=0.1, 
                 activacion_oculta='relu', activacion_salida='sigmoid'):
        print(f"🧠 ENTRENANDO CEREBRO NUMPY")
        print(f"   Capas: {self.capas}")
        print(f"   Activación oculta: {activacion_oculta}")
        print(f"   Activación salida: {activacion_salida}")
        
        for epoch in range(epochs):
            salida = self.forward(X, activacion_oculta, activacion_salida)
            error = self.backward(X, y, salida, lr)
            if epoch % 2000 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        
        return error
    
    def predecir(self, X, activacion_oculta='relu', activacion_salida='sigmoid'):
        return self.forward(X, activacion_oculta, activacion_salida)
    
    def guardar(self, nombre="cerebro_numpy.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos,
                'sesgos': self.sesgos,
                'capas': self.capas
            }, f)
        print(f"💾 Guardado en {nombre}")
    
    def cargar(self, nombre="cerebro_numpy.pkl"):
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.pesos = data['pesos']
        self.sesgos = data['sesgos']
        self.capas = data['capas']
        print(f"📂 Cargado desde {nombre}")

def prueba_cerebro_numpy():
    print("🧠 PROBANDO CEREBRO NUMPY")
    print("="*40)
    
    # XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)
    
    cerebro = CerebroNumpy([2, 16, 16, 1])
    cerebro.entrenar(X, y, epochs=10000, lr=0.5)
    
    print("\n✅ RESULTADOS:")
    for i, x in enumerate(X):
        pred = cerebro.predecir(x.reshape(1, -1))[0, 0]
        print(f"   {x} → {pred:.4f}")
    
    cerebro.guardar("cerebro_numpy.pkl")

if __name__ == "__main__":
    prueba_cerebro_numpy()
