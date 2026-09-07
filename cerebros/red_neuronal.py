import numpy as np
import pickle

class Cerebro:
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
            # Usar EXACTAMENTE la misma inicialización que la opción 1
            w = np.random.randn(capas[i], capas[i+1]) * 0.1
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, entrada):
        self.activaciones = [np.array(entrada)]
        for w, b in zip(self.pesos, self.sesgos):
            z = np.dot(self.activaciones[-1], w) + b
            a = self.sigmoid(z)
            self.activaciones.append(a)
        return self.activaciones[-1]
    
    def predecir(self, entrada):
        return self.forward(entrada)
    
    def guardar(self, nombre="cerebro.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({'pesos': self.pesos, 'sesgos': self.sesgos, 'capas': self.capas}, f)
        print(f"🧠 Guardado en {nombre}")
    
    def cargar(self, nombre="cerebro.pkl"):
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.pesos = data['pesos']
        self.sesgos = data['sesgos']
        self.capas = data['capas']
        print(f"🧠 Cargado desde {nombre}")
    
    def resumen(self):
        print(f"🧠 ESTRUCTURA DEL CEREBRO:")
        print(f"   Capas: {self.capas}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params}")
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            print(f"   Capa {i+1}: {w.shape} pesos + {b.shape} sesgos")
