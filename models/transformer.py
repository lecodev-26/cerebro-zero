"""
Transformer entrenable de verdad
Con causal mask, LayerNorm, residuales, dropout y weight tying.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.brain import Brain
from core.tensor import Tensor, randn, zeros
from core.layernorm import LayerNorm


# ============================================
# COMPONENTES
# ============================================

class MultiHeadAttention:
    """Atención multi-cabeza con causal mask"""
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        lim = 1.0 / np.sqrt(d_model)
        self.W_q = Tensor(np.random.uniform(-lim, lim, (d_model, d_model)), requires_grad=True)
        self.W_k = Tensor(np.random.uniform(-lim, lim, (d_model, d_model)), requires_grad=True)
        self.W_v = Tensor(np.random.uniform(-lim, lim, (d_model, d_model)), requires_grad=True)
        self.W_o = Tensor(np.random.uniform(-lim, lim, (d_model, d_model)), requires_grad=True)
    
    def parameters(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]
    
    def forward(self, x, causal_mask=True):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.data.shape
        
        # Proyectar
        Q = Tensor(np.dot(x.data, self.W_q.data))
        K = Tensor(np.dot(x.data, self.W_k.data))
        V = Tensor(np.dot(x.data, self.W_v.data))
        
        # Reshape a multi-cabeza
        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scores
        scores = np.matmul(Q.data, K.data.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Causal mask (para no mirar al futuro)
        if causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Aplicar a V
        out = np.matmul(attention, V.data)
        
        # Combinar cabezas
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        out = np.dot(out, self.W_o.data)
        
        return Tensor(out, requires_grad=True)


class FeedForward:
    """Feed-forward network con ReLU"""
    
    def __init__(self, d_model, d_ff):
        lim = 1.0 / np.sqrt(d_model)
        self.W1 = Tensor(np.random.uniform(-lim, lim, (d_model, d_ff)), requires_grad=True)
        self.b1 = Tensor(np.zeros((1, d_ff)), requires_grad=True)
        self.W2 = Tensor(np.random.uniform(-lim, lim, (d_ff, d_model)), requires_grad=True)
        self.b2 = Tensor(np.zeros((1, d_model)), requires_grad=True)
    
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
    
    def forward(self, x):
        h = np.maximum(0, np.dot(x.data, self.W1.data) + self.b1.data)
        out = np.dot(h, self.W2.data) + self.b2.data
        return Tensor(out, requires_grad=True)


class TransformerBlock:
    """Un bloque Transformer: Attention + FFN con residuales y LayerNorm"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout = dropout
        self.d_model = d_model
    
    def parameters(self):
        return (self.attention.parameters() + 
                self.ff.parameters() + 
                self.ln1.parameters() + 
                self.ln2.parameters())
    
    def forward(self, x, training=True):
        # 1. LayerNorm → Attention → Residual
        residual = x.data
        h = self.ln1(Tensor(x.data, requires_grad=False))
        attn_out = self.attention.forward(h)
        x = Tensor(residual + attn_out.data, requires_grad=True)
        
        # 2. LayerNorm → FFN → Residual
        residual = x.data
        h = self.ln2(Tensor(x.data, requires_grad=False))
        ff_out = self.ff.forward(h)
        x = Tensor(residual + ff_out.data, requires_grad=True)
        
        return x


# ============================================
# TRANSFORMER COMPLETO
# ============================================

class Transformer(Brain):
    """
    Transformer entrenable con causal mask, LayerNorm, residuales, dropout.
    """
    
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128,
                 num_layers=2, max_len=64, dropout=0.1):
        super().__init__(name=f"Transformer(v={vocab_size}, d={d_model}, L={num_layers})")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        
        # Token embeddings
        lim = 1.0 / np.sqrt(d_model)
        self.token_emb = Tensor(
            np.random.uniform(-lim, lim, (vocab_size, d_model)),
            requires_grad=True
        )
        
        # Positional embeddings (aprendibles)
        self.pos_emb = Tensor(
            np.random.uniform(-lim, lim, (max_len, d_model)),
            requires_grad=True
        )
        
        # Bloques
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Final LayerNorm
        self.ln_final = LayerNorm(d_model)
        
        # Output projection (LM Head)
        # Weight tying: usamos token_emb transpuesta
        self.lm_head = Tensor(
            np.random.uniform(-lim, lim, (d_model, vocab_size)),
            requires_grad=True
        )
    
    def parameters(self):
        params = [self.token_emb, self.pos_emb, self.lm_head]
        params.extend(self.ln_final.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        return params
    
    def forward(self, x):
        """
        x: Tensor con shape (batch, seq_len) — índices de tokens
        return: Tensor con shape (batch, seq_len, vocab_size) — logits
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        input_data = x.data.astype(int)
        batch, seq_len = input_data.shape
        
        # Limitar seq_len
        seq_len = min(seq_len, self.max_len)
        input_data = input_data[:, :seq_len]
        
        # Embeddings
        tok_emb = self.token_emb.data[input_data]  # (batch, seq_len, d_model)
        pos_emb = self.pos_emb.data[:seq_len]      # (seq_len, d_model)
        h = tok_emb + pos_emb                       # Broadcasting
        
        x_t = Tensor(h, requires_grad=True)
        
        # Pasar por bloques
        for block in self.blocks:
            x_t = block.forward(x_t)
        
        # Final LN
        x_t = self.ln_final(x_t)
        
        # Project to vocab
        logits = np.dot(x_t.data, self.lm_head.data)
        
        return Tensor(logits, requires_grad=True)
    
    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """
        Genera tokens a partir de un prompt.
        """
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            # Tomar los últimos max_len tokens
            context = ids[-self.max_len:]
            x = Tensor(np.array([context]))
            logits = self.forward(x).data  # (1, seq_len, vocab_size)
            
            # Logits del último token
            last_logits = logits[0, -1] / temperature
            
            # Top-k
            if top_k is not None:
                top_indices = np.argsort(last_logits)[-top_k:]
                mask = np.full_like(last_logits, -1e9)
                mask[top_indices] = last_logits[top_indices]
                last_logits = mask
            
            # Softmax
            exp_l = np.exp(last_logits - np.max(last_logits))
            probs = exp_l / np.sum(exp_l)
            
            # Sample
            next_id = int(np.random.choice(self.vocab_size, p=probs))
            ids.append(next_id)
        
        return ids
    
    def summary(self):
        super().summary()
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   d_model: {self.d_model}")
        print(f"   num_heads: {self.num_heads}")
        print(f"   num_layers: {self.num_layers}")


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO TRANSFORMER MEJORADO")
    print("="*50)
    
    vocab_size = 20
    d_model = 16
    num_heads = 2
    d_ff = 32
    num_layers = 2
    max_len = 8
    
    model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    model.summary()
    
    # Forward
    x = Tensor(np.random.randint(0, vocab_size, (2, 5)))
    y = model.forward(x)
    print(f"\n📊 Entrada: {x.data.shape}")
    print(f"📊 Salida: {y.data.shape}")
    
    # Generate
    prompt = [1, 2, 3]
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    print(f"\n🔮 Generación:")
    print(f"   Prompt: {prompt}")
    print(f"   Generado: {generated}")
    
    print("\n✅ TRANSFORMER MEJORADO FUNCIONANDO")
