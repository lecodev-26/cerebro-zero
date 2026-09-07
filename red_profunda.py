import numpy as np
import pickle

class CerebroProfundo:
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
            w = np.random.randn(capas[i], capas[i+1]) * np.sqrt(2.0 / capas[i])
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, entrada, entrenando=False):
        self.activaciones = [np.array(entrada)]
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
    
    def predecir(self, entrada):
        return self.forward(entrada)
    
    def resumen(self):
        print(f"🧠 CEREBRO PROFUNDO")
        print(f"   Capas: {self.capas}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params}")
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            tipo = "ReLU" if i < len(self.pesos)-1 else "Sigmoid"
            print(f"   Capa {i+1}: {w.shape} → {b.shape} (activación: {tipo})")
    
    def guardar(self, nombre="profundo_xor.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({'pesos': self.pesos, 'sesgos': self.sesgos, 'capas': self.capas}, f)
        print(f"💾 Guardado en {nombre}")
    
    def cargar(self, nombre="profundo_xor.pkl"):
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.pesos = data['pesos']
        self.sesgos = data['sesgos']
        self.capas = data['capas']
        print(f"📂 Cargado desde {nombre}")
