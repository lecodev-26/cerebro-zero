import numpy as np
from red_neuronal import Cerebro

class CerebroConMemoria:
    def __init__(self, entrada_size, memoria_size=3):
        self.memoria_size = memoria_size
        self.memoria = np.zeros((memoria_size, entrada_size))
        self.posicion = 0
        
        entrada_total = entrada_size + (memoria_size * entrada_size)
        # Más neuronas para aprender mejor
        self.cerebro = Cerebro([entrada_total, 32, 1])
    
    def recordar(self, entrada):
        self.memoria[self.posicion] = entrada
        self.posicion = (self.posicion + 1) % self.memoria_size
    
    def predecir(self, entrada):
        self.recordar(entrada)
        entrada_completa = np.concatenate([entrada, self.memoria.flatten()])
        return self.cerebro.predecir(entrada_completa)
    
    def entrenar(self, X, y, epochs=2000):
        from entrenar import Entrenador
        entrenador = Entrenador(self.cerebro, tasa=0.3)  # Tasa más alta
        
        X_seq = []
        y_seq = []
        for i in range(self.memoria_size, len(X)):
            seq_X = np.concatenate([X[i], X[i-self.memoria_size:i].flatten()])
            seq_y = y[i]
            X_seq.append(seq_X)
            y_seq.append(seq_y)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"🧠 Entrenando con memoria (secuencias de {self.memoria_size})")
        for epoch in range(epochs):
            error = entrenador.entrenar_epoch(X_seq, y_seq)
            if epoch % 400 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        
        return error
    
    def guardar(self, nombre="memoria.pkl"):
        import pickle
        with open(nombre, 'wb') as f:
            pickle.dump({
                'cerebro': self.cerebro,
                'memoria': self.memoria,
                'posicion': self.posicion,
                'memoria_size': self.memoria_size
            }, f)
        print(f"💾 Cerebro con memoria guardado en {nombre}")
    
    def cargar(self, nombre="memoria.pkl"):
        import pickle
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.cerebro = data['cerebro']
        self.memoria = data['memoria']
        self.posicion = data['posicion']
        self.memoria_size = data['memoria_size']
        print(f"📂 Cerebro con memoria cargado desde {nombre}")
