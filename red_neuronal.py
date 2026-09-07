import numpy as np

class Cerebro:
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
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
    
    def guardar(self, nombre="cerebro.npy"):
        np.savez(nombre, pesos=self.pesos, sesgos=self.sesgos)
        print(f"🧠 Guardado en {nombre}")
    
    def cargar(self, nombre="cerebro.npy"):
        data = np.load(nombre, allow_pickle=True)
        self.pesos = data['pesos']
        self.sesgos = data['sesgos']
        print(f"🧠 Cargado desde {nombre}")
