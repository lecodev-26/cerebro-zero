import sys
sys.path.append('..')
import numpy as np
import pickle

class CerebroAXIOM:
    def __init__(self, capas, entropia_umbral=0.5, tasa_aprendizaje=0.5):
        self.capas = capas
        self.entropia_umbral = entropia_umbral
        self.tasa_aprendizaje = tasa_aprendizaje
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
            lim = np.sqrt(6.0 / (capas[i] + capas[i+1]))
            w = np.random.uniform(-lim, lim, (capas[i], capas[i+1]))
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def shannon_entropy(self, x):
        p = np.abs(x) / (np.sum(np.abs(x)) + 1e-10)
        return -np.sum(p * np.log2(p + 1e-10))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivada(self, x):
        return x * (1 - x)
    
    def forward(self, X, entrenando=False):
        self.activaciones = [X]
        self.lineales = []
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = np.dot(self.activaciones[-1], w) + b
            self.lineales.append(z)
            
            if entrenando and i < len(self.pesos) - 1:
                entropia = self.shannon_entropy(z)
                if entropia > self.entropia_umbral:
                    z = z * 0.95
            
            if i == len(self.pesos) - 1:
                a = self.sigmoid(z)
            else:
                a = np.maximum(0, z)
            
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def backward(self, X, y, salida):
        error = y - salida
        grad = error * self.sigmoid_derivada(salida)
        
        for i in range(len(self.pesos)-1, -1, -1):
            self.pesos[i] += self.tasa_aprendizaje * np.dot(self.activaciones[i].T, grad)
            self.sesgos[i] += self.tasa_aprendizaje * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.pesos[i].T)
                grad = grad * (self.lineales[i-1] > 0).astype(float)
        
        return np.mean(error ** 2)
    
    def entrenar(self, X, y, epochs=10000):
        print("🏋️ ENTRENANDO CEREBRO AXIOM...")
        for epoch in range(epochs):
            salida = self.forward(X, entrenando=True)
            error = self.backward(X, y, salida)
            if epoch % 1000 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        return error
    
    def predecir(self, X):
        return self.forward(X, entrenando=False)
    
    def guardar(self, nombre="cerebro_axiom.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos,
                'sesgos': self.sesgos,
                'capas': self.capas,
                'entropia_umbral': self.entropia_umbral
            }, f)
        print(f"💾 Cerebro AXIOM guardado en {nombre}")
    
    def resumen(self):
        print("🧠 CEREBRO AXIOM (Basado en AXION)")
        print("="*40)
        print(f"   Capas: {self.capas}")
        print(f"   Entropía umbral: {self.entropia_umbral}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params:,}")
        print(f"   Tasa de aprendizaje: {self.tasa_aprendizaje}")
