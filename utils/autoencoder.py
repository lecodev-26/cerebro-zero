import numpy as np

class Autoencoder:
    def __init__(self, entrada_size, codificado_size=5):
        self.entrada_size = entrada_size
        self.codificado_size = codificado_size
        
        from red_neuronal import Cerebro
        self.encoder = Cerebro([entrada_size, codificado_size*2, codificado_size])
        self.decoder = Cerebro([codificado_size, codificado_size*2, entrada_size])
    
    def codificar(self, X):
        return self.encoder.predecir(X)
    
    def decodificar(self, codificado):
        return self.decoder.predecir(codificado)
    
    def reconstruir(self, X):
        codificado = self.codificar(X)
        return self.decodificar(codificado)
    
    def entrenar(self, X, epochs=1000):
        from entrenar import Entrenador
        
        print(f"📦 ENTRENANDO AUTOENCODER")
        print(f"   {self.entrada_size} → {self.codificado_size} → {self.entrada_size}")
        
        for epoch in range(epochs):
            codificado = self.encoder.forward(X)
            reconstruido = self.decoder.forward(codificado)
            error = np.mean((X - reconstruido) ** 2)
            
            grad = (reconstruido - X) * reconstruido * (1 - reconstruido)
            for i in range(len(self.decoder.pesos)-1, -1, -1):
                if i > 0:
                    grad = np.dot(grad, self.decoder.pesos[i].T)
                    grad = grad * self.decoder.activaciones[i] * (1 - self.decoder.activaciones[i])
                self.decoder.pesos[i] += 0.3 * np.dot(self.decoder.activaciones[i].T, grad)
                self.decoder.sesgos[i] += 0.3 * np.sum(grad, axis=0, keepdims=True)
            
            grad = (reconstruido - X) * reconstruido * (1 - reconstruido)
            for i in range(len(self.encoder.pesos)-1, -1, -1):
                if i > 0:
                    grad = np.dot(grad, self.encoder.pesos[i].T)
                    grad = grad * self.encoder.activaciones[i] * (1 - self.encoder.activaciones[i])
                self.encoder.pesos[i] += 0.3 * np.dot(self.encoder.activaciones[i].T, grad)
                self.encoder.sesgos[i] += 0.3 * np.sum(grad, axis=0, keepdims=True)
            
            if epoch % 200 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        
        return error
    
    def guardar(self, nombre="autoencoder.npy"):
        np.savez(nombre, 
                 enc_pesos=self.encoder.pesos, 
                 enc_sesgos=self.encoder.sesgos,
                 dec_pesos=self.decoder.pesos, 
                 dec_sesgos=self.decoder.sesgos)
        print(f"💾 Autoencoder guardado en {nombre}")
