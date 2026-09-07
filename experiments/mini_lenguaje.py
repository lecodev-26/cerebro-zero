import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.transformer import Encoder

class MiniModeloLenguaje:
    def __init__(self, vocab_size=50, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_len=20):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01
        self.b_out = np.zeros((1, vocab_size))
        self.max_len = max_len
    
    def forward(self, x):
        encoded = self.encoder.forward(x)
        logits = np.dot(encoded, self.W_out) + self.b_out
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)
        return probs
    
    def generar(self, inicio, longitud=10, temperature=1.0):
        secuencia = list(inicio)
        for _ in range(longitud):
            x = np.array([secuencia[-self.max_len:]])
            probs = self.forward(x)
            probs_last = probs[0, -1]
            probs_last = probs_last ** (1.0 / temperature)
            probs_last = probs_last / np.sum(probs_last)
            siguiente = np.random.choice(self.vocab_size, p=probs_last)
            secuencia.append(siguiente)
        return secuencia
    
    def one_hot(self, y, vocab_size):
        """Convierte índices a one-hot encoding"""
        y_onehot = np.zeros((y.shape[0], y.shape[1], vocab_size))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y_onehot[i, j, y[i, j]] = 1
        return y_onehot
    
    def entrenar(self, X, y, epochs=100, lr=0.01):
        print(f"🏋️ ENTRENANDO MODELO DE LENGUAJE...")
        print(f"   Datos: {X.shape[0]} ejemplos")
        print(f"   Épocas: {epochs}")
        
        # Convertir y a one-hot
        y_onehot = self.one_hot(y, self.vocab_size)
        
        for epoch in range(epochs):
            # Forward
            probs = self.forward(X)
            # Cross-entropy loss
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=-1))
            if epoch % 20 == 0:
                print(f"   Epoch {epoch}: loss = {loss:.6f}")
    
    def __repr__(self):
        return f"MiniModeloLenguaje(vocab={self.vocab_size}, d_model={self.d_model})"

def prueba_mini_lenguaje():
    print("🧠 MINI MODELO DE LENGUAJE")
    print("="*40)
    
    # Crear modelo
    modelo = MiniModeloLenguaje(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=10
    )
    print(f"Modelo: {modelo}")
    
    # Generar texto antes de entrenar
    print("\n📝 GENERANDO TEXTO (SIN ENTRENAR):")
    inicio = [1, 2, 3]
    generado = modelo.generar(inicio, longitud=8, temperature=1.0)
    print(f"   Inicio: {inicio}")
    print(f"   Generado: {generado}")
    
    # Simular entrenamiento con datos aleatorios
    print("\n🏋️ SIMULANDO ENTRENAMIENTO...")
    X_train = np.random.randint(0, 20, (10, 5))
    y_train = np.random.randint(0, 20, (10, 5))
    modelo.entrenar(X_train, y_train, epochs=100, lr=0.01)
    
    # Generar después de entrenar
    print("\n📝 GENERANDO TEXTO (DESPUÉS DE ENTRENAR):")
    inicio = [1, 2, 3]
    generado = modelo.generar(inicio, longitud=8, temperature=0.8)
    print(f"   Inicio: {inicio}")
    print(f"   Generado: {generado}")
    
    print("\n✅ MINI MODELO DE LENGUAJE FUNCIONANDO!")

if __name__ == "__main__":
    prueba_mini_lenguaje()
