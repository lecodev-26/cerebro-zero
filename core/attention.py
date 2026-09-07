import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Pesos para Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, Q, K, V):
        # Proyectar
        Q = np.dot(Q, self.W_q)
        K = np.dot(K, self.W_k)
        V = np.dot(V, self.W_v)
        
        # Dividir en cabezas
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Atención escalada
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention = self.softmax(scores)
        output = np.matmul(attention, V)
        
        # Combinar cabezas
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(output, self.W_o)
        
        return output, attention
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)
    
    def __repr__(self):
        return f"MultiHeadAttention(d_model={self.d_model}, heads={self.num_heads})"
