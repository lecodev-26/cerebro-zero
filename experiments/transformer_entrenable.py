import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.transformer import Encoder

class TransformerEntrenable:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=50):
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Capa de salida
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01
        self.b_out = np.zeros((1, vocab_size))
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder.forward(x)
        # Proyección a vocabulario
        logits = np.dot(encoded, self.W_out) + self.b_out
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)
        return probs
    
    def predecir(self, x, temperature=1.0):
        probs = self.forward(x)
        # Aplicar temperatura
        probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        # Muestrear
        return np.random.choice(self.vocab_size, p=probs[0, -1])
    
    def __repr__(self):
        return f"TransformerEntrenable(vocab={self.vocab_size}, d_model={self.d_model})"

def prueba_transformer_entrenable():
    print("🧠 TRANSFORMER ENTRENABLE")
    print("="*40)
    
    # Crear modelo
    modelo = TransformerEntrenable(
        vocab_size=50,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=2,
        max_len=20
    )
    print(f"Modelo: {modelo}")
    
    # Generar datos de ejemplo (secuencias aleatorias)
    X = np.random.randint(0, 50, (4, 10))
    print(f"\nEntrada: {X.shape}")
    
    # Forward
    output = modelo.forward(X)
    print(f"Salida: {output.shape}")
    
    print("\n✅ TRANSFORMER ENTRENABLE FUNCIONANDO!")
    
    # Probar predicción
    print("\n🔮 Predicción de muestra:")
    x_test = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    pred = modelo.predecir(x_test)
    print(f"   Entrada: {x_test[0]}")
    print(f"   Predicción: {pred}")

if __name__ == "__main__":
    prueba_transformer_entrenable()
