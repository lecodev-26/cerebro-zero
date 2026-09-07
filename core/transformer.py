import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.attention import MultiHeadAttention

class PositionalEncoding:
    def __init__(self, d_model, max_len=100):
        self.d_model = d_model
        self.max_len = max_len
        self.encoding = self._crear_encoding()
    
    def _crear_encoding(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def forward(self, x):
        return x + self.encoding[:x.shape[1], :]

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))
    
    def forward(self, x):
        return np.dot(np.maximum(0, np.dot(x, self.W1) + self.b1), self.W2) + self.b2
    
    def __repr__(self):
        return f"FeedForward(d_model={self.W1.shape[0]}, d_ff={self.W1.shape[1]})"

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = dropout
    
    def forward(self, x):
        # Multi-head attention con residual
        attn_output, _ = self.attention.forward(x, x, x)
        x = x + attn_output
        # Feed-forward con residual
        ff_output = self.ff.forward(x)
        x = x + ff_output
        return x
    
    def __repr__(self):
        return f"EncoderLayer({self.attention}, {self.ff})"

class Encoder:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=100):
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x):
        # Embedding + positional encoding
        x = self.embedding[x] + self.pos_encoding.encoding[:x.shape[1], :]
        # Pasar por todas las capas
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def __repr__(self):
        return f"Encoder(vocab={self.embedding.shape[0]}, d_model={self.d_model}, layers={len(self.layers)})"

def prueba_transformer():
    print("🧠 PROBANDO TRANSFORMER")
    print("="*40)
    
    # Crear encoder pequeño
    vocab_size = 100
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2
    max_len = 10
    
    encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    print(f"Encoder: {encoder}")
    
    # Datos de prueba
    X = np.random.randint(0, vocab_size, (2, 5))  # Batch de 2, secuencia de 5
    print(f"\nEntrada: {X.shape}")
    
    # Forward
    output = encoder.forward(X)
    print(f"Salida: {output.shape}")
    
    print("\n✅ TRANSFORMER FUNCIONANDO CORRECTAMENTE!")
    return encoder

if __name__ == "__main__":
    prueba_transformer()
