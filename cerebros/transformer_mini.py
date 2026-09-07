import sys
sys.path.append('..')
import numpy as np
import pickle

class TransformerMini:
    """
    Transformer en miniatura para móvil
    - Atención multi-cabeza (2 cabezas)
    - 2 capas de atención
    - Feed-forward con ReLU
    """
    def __init__(self, vocab_size=100, d_model=32, num_heads=2, d_ff=64, num_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        
        # Pesos de atención por capa
        self.W_q = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        self.W_k = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        self.W_v = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        self.W_o = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        
        # Feed-forward por capa
        self.W_ff1 = [np.random.randn(d_model, d_ff) * 0.01 for _ in range(num_layers)]
        self.W_ff2 = [np.random.randn(d_ff, d_model) * 0.01 for _ in range(num_layers)]
        
        # Capa final
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01
        
        # Parámetros totales
        total_params = (
            vocab_size * d_model +  # embedding
            num_layers * (4 * d_model * d_model + 2 * d_model * d_ff) +  # atención + FF
            d_model * vocab_size  # salida
        )
        print(f"🧠 TRANSFORMER MINI CREADO")
        print(f"   Vocabulario: {vocab_size}")
        print(f"   Dimensiones: {d_model}")
        print(f"   Cabezas: {num_heads}")
        print(f"   Capas: {num_layers}")
        print(f"   Parámetros: {total_params:,}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def attention(self, Q, K, V):
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        weights = self.softmax(scores)
        return np.dot(weights, V)
    
    def forward(self, x):
        # Embedding
        x = self.embedding[x]
        
        for layer in range(self.num_layers):
            # Multi-head attention
            Q = np.dot(x, self.W_q[layer])
            K = np.dot(x, self.W_k[layer])
            V = np.dot(x, self.W_v[layer])
            
            # Dividir en cabezas
            head_size = self.d_model // self.num_heads
            Q_heads = Q.reshape(-1, self.num_heads, head_size)
            K_heads = K.reshape(-1, self.num_heads, head_size)
            V_heads = V.reshape(-1, self.num_heads, head_size)
            
            # Atención por cabeza
            attn_heads = []
            for h in range(self.num_heads):
                attn = self.attention(Q_heads[:, h], K_heads[:, h], V_heads[:, h])
                attn_heads.append(attn)
            
            # Combinar cabezas
            x = np.concatenate(attn_heads, axis=-1)
            x = np.dot(x, self.W_o[layer])
            
            # Feed-forward
            ff = self.relu(np.dot(x, self.W_ff1[layer]))
            ff = np.dot(ff, self.W_ff2[layer])
            x = x + ff  # Residual
        
        # Salida
        logits = np.dot(x, self.W_out)
        return self.softmax(logits)
    
    def predecir(self, x):
        return self.forward(x)
    
    def guardar(self, nombre="transformer_mini.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({
                'embedding': self.embedding,
                'W_q': self.W_q,
                'W_k': self.W_k,
                'W_v': self.W_v,
                'W_o': self.W_o,
                'W_ff1': self.W_ff1,
                'W_ff2': self.W_ff2,
                'W_out': self.W_out,
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'd_ff': self.d_ff,
                'num_layers': self.num_layers
            }, f)
        print(f"💾 Guardado en {nombre}")
