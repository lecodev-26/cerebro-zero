import numpy as np
from red_neuronal import Cerebro

class CerebroConMemoria:
    def __init__(self, entrada_size, memoria_size=10):
        self.memoria_size = memoria_size
        self.memoria = np.zeros((memoria_size, entrada_size))
        self.posicion = 0
        
        # Cerebro que procesa entradas + memoria
        self.cerebro = Cerebro([entrada_size + memoria_size, 20, 1])
    
    def recordar(self, entrada):
        """Añade entrada a la memoria (circular)"""
        self.memoria[self.posicion] = entrada
        self.posicion = (self.posicion + 1) % self.memoria_size
    
    def predecir(self, entrada):
        """Predice usando entrada + memoria"""
        self.recordar(entrada)
        # Usar la última entrada más la memoria completa
        entrada_completa = np.concatenate([entrada, self.memoria.flatten()])
        return self.cerebro.predecir(entrada_completa)
    
    def entrenar(self, X, y, epochs=1000):
        """Entrena con secuencias"""
        from entrenar import Entrenador
        entrenador = Entrenador(self.cerebro, tasa=0.5)
        
        # Convertir datos en secuencias
        X_seq = []
        y_seq = []
        for i in range(len(X) - self.memoria_size):
            # Usar los últimos memoria_size+1 puntos
            seq_X = X[i:i+self.memoria_size].flatten()
            seq_y = y[i+self.memoria_size]
            X_seq.append(seq_X)
            y_seq.append(seq_y)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"🧠 Entrenando con memoria (secuencias de {self.memoria_size})")
        for epoch in range(epochs):
            error = entrenador.entrenar_epoch(X_seq, y_seq)
            if epoch % 200 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        
        return error
    
    def guardar(self, nombre="memoria.npy"):
        np.savez(nombre, 
                 pesos=self.cerebro.pesos, 
                 sesgos=self.cerebro.sesgos,
                 memoria=self.memoria,
                 posicion=self.posicion)
        print(f"💾 Cerebro con memoria guardado en {nombre}")
    
    def cargar(self, nombre="memoria.npy"):
        data = np.load(nombre, allow_pickle=True)
        self.cerebro.pesos = data['pesos']
        self.cerebro.sesgos = data['sesgos']
        self.memoria = data['memoria']
        self.posicion = data['posicion']
        print(f"📂 Cerebro con memoria cargado desde {nombre}")
