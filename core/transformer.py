"""
Transformer desde cero - Corregido para guardado
"""

import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape
        
        Q = np.dot(Q, self.W_q)
        K = np.dot(K, self.W_k)
        V = np.dot(V, self.W_v)
        
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask
        
        scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = scores / (np.sum(scores, axis=-1, keepdims=True) + 1e-10)
        
        output = np.matmul(attention, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(output, self.W_o)
        
        return output, attention
    
    def __getstate__(self):
        # Para pickle: devolver solo atributos serializables
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_k': self.d_k,
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)

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
    
    def __getstate__(self):
        return {
            'd_model': self.d_model,
            'max_len': self.max_len,
            'encoding': self.encoding
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))
    
    def forward(self, x):
        return np.dot(np.maximum(0, np.dot(x, self.W1) + self.b1), self.W2) + self.b2
    
    def __getstate__(self):
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        # Usamos funciones normales en lugar de lambdas para pickle
        self._ln1 = None
        self._ln2 = None
    
    def forward(self, x, mask=None):
        attn_output, _ = self.attention.forward(x, x, x, mask)
        x = x + attn_output
        ff_output = self.ff.forward(x)
        x = x + ff_output
        return x
    
    def __getstate__(self):
        return {
            'attention': self.attention,
            'ff': self.ff
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)

class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=100):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.W_out = np.random.randn(d_model, vocab_size) * 0.01
        self.b_out = np.zeros((1, vocab_size))
    
    def forward(self, x, mask=None):
        x = self.embedding[x] + self.pos_encoding.encoding[:x.shape[1], :]
        for layer in self.layers:
            x = layer.forward(x, mask)
        logits = np.dot(x, self.W_out) + self.b_out
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)
        return probs
    
    def predict(self, x, temperature=1.0):
        probs = self.forward(x)
        probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        return np.random.choice(self.vocab_size, p=probs[0, -1])
    
    def __getstate__(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'max_len': self.max_len,
            'embedding': self.embedding,
            'pos_encoding': self.pos_encoding,
            'layers': self.layers,
            'W_out': self.W_out,
            'b_out': self.b_out
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)

def prueba_transformer():
    print("🧠 PROBANDO TRANSFORMER")
    print("="*30)
    
    vocab_size = 50
    d_model = 16
    num_heads = 2
    d_ff = 32
    num_layers = 2
    max_len = 20
    
    transformer = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    
    x = np.random.randint(0, vocab_size, (2, 5))
    probs = transformer.forward(x)
    
    print(f"📊 Entrada: {x.shape}")
    print(f"📊 Salida: {probs.shape}")
    print(f"📊 Vocabulario: {vocab_size}")
    print(f"📊 Dimensiones: {d_model}")
    print(f"📊 Capas: {num_layers}")
    print(f"📊 Cabezas: {num_heads}")
    
    pred = transformer.predict(x)
    print(f"🔮 Predicción: {pred}")
    
    print("\n✅ TRANSFORMER FUNCIONANDO!")

if __name__ == "__main__":
    prueba_transformer()
