import sys
sys.path.append('..')
import numpy as np
import pickle

class CerebroGigante:
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
            # Inicialización más pequeña para evitar explosión
            w = np.random.randn(capas[i], capas[i+1]) * np.sqrt(1.0 / capas[i])  # Xavier
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
        
        print(f"🧠 CEREBRO GIGANTE CREADO")
        print(f"   Capas: {capas}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params:,}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activaciones = [X]
        self.lineales = []
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = np.dot(self.activaciones[-1], w) + b
            # Recortar para evitar overflow
            z = np.clip(z, -10, 10)
            self.lineales.append(z)
            if i == len(self.pesos) - 1:
                a = z  # Salida lineal
            else:
                a = self.relu(z)
                # Recortar también las activaciones
                a = np.clip(a, 0, 10)
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def predecir(self, X):
        return self.forward(X)
    
    def guardar(self, nombre="cerebro_gigante.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({'pesos': self.pesos, 'sesgos': self.sesgos, 'capas': self.capas}, f)
        print(f"💾 Guardado en {nombre}")
    
    def cargar(self, nombre="cerebro_gigante.pkl"):
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.pesos = data['pesos']
        self.sesgos = data['sesgos']
        self.capas = data['capas']
        print(f"📂 Cargado desde {nombre}")
