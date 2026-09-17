"""
Transformer con backprop REAL de extremo a extremo.
Todo dentro del grafo de autograd. Sin cortes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.brain import Brain
from core.tensor import Tensor
from core.layernorm import LayerNorm


# ============================================
# MULTI-HEAD ATTENTION (autograd completo)
# ============================================

class MultiHeadAttention:
    """
    Multi-Head Attention con autograd completo.
    
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    scores = Q @ K^T / sqrt(d_k)
    attention = softmax(scores)
    out = attention @ V
    out = out @ W_o
    """
    
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
        x: Tensor (batch, seq_len, d_model)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        batch, seq_len, _ = x.data.shape
        
        # Proyecciones (todo dentro del grafo)
        Q = x.batch_matmul(self.W_q)  # (batch, seq_len, d_model)
        K = x.batch_matmul(self.W_k)
        V = x.batch_matmul(self.W_v)
        
        # Reshape a (batch, seq_len, num_heads, d_k)
        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k)
        
        # Transpose a (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scores = Q @ K^T
        K_T = K.transpose(0, 1, 3, 2)
        scores = Q.batch_matmul(K_T)  # (batch, num_heads, seq_len, seq_len)
        scores = scores * (1.0 / np.sqrt(self.d_k))
        
        # Causal mask (para no mirar al futuro)
        if causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            mask = (1 - mask)  # 1 donde SÍ puede mirar, 0 donde NO
            # Aplicar máscara: scores en posiciones futuras → -inf → softmax = 0
            mask_broadcast = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, seq)
            # Usar where para poner -1e9 donde mask=0
            big_neg = np.full_like(scores.data, -1e9)
            scores = scores.where(mask_broadcast > 0.5, Tensor(big_neg, requires_grad=False))
        
        # Softmax sobre la última dimensión
        attention = scores.softmax(axis=-1)
        
        # Aplicar a V: out = attention @ V
        out = attention.batch_matmul(V)  # (batch, num_heads, seq_len, d_k)
        
        # Combinar cabezas: (batch, seq_len, d_model)
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch, seq_len, self.d_model)
        
        # Proyección final
        out = out.batch_matmul(self.W_o)
        
        return out
    
    def __call__(self, x, causal_mask=True):
        return self.forward(x, causal_mask)


# ============================================
# FEED FORWARD (autograd completo)
# ============================================

class FeedForward:
    """
    Feed Forward con autograd completo.
    
    out = ReLU(x @ W1 + b1) @ W2 + b2
    """
    
    def __init__(self, d_model, d_ff):
        lim = 1.0 / np.sqrt(d_model)
        self.W1 = Tensor(np.random.uniform(-lim, lim, (d_model, d_ff)), requires_grad=True)
        self.b1 = Tensor(np.zeros((d_ff,)), requires_grad=True)
        self.W2 = Tensor(np.random.uniform(-lim, lim, (d_ff, d_model)), requires_grad=True)
        self.b2 = Tensor(np.zeros((d_model,)), requires_grad=True)
    
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
    
    def forward(self, x):
        """
        x: Tensor (batch, seq_len, d_model)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        h = x.batch_matmul(self.W1)  # (batch, seq_len, d_ff)
        h = h + self.b1
        h = h.relu()
        out = h.batch_matmul(self.W2)  # (batch, seq_len, d_model)
        out = out + self.b2
        return out
    
    def __call__(self, x):
        return self.forward(x)


# ============================================
# TRANSFORMER BLOCK (autograd completo)
# ============================================

class TransformerBlock:
    """Un bloque Transformer con residuales y LayerNorm"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
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
    
    def forward(self, x, causal_mask=True):
        # 1. LayerNorm → Attention → Residual
        h = self.ln1.forward(x)
        attn_out = self.attention.forward(h, causal_mask=causal_mask)
        x = x + attn_out  # Residual
        
        # 2. LayerNorm → FFN → Residual
        h = self.ln2.forward(x)
        ff_out = self.ff.forward(h)
        x = x + ff_out  # Residual
        
        return x
    
    def __call__(self, x, causal_mask=True):
        return self.forward(x, causal_mask)


# ============================================
# TRANSFORMER COMPLETO
# ============================================

class Transformer(Brain):
    """
    Transformer con autograd completo.
    
    Input: (batch, seq_len) - índices de tokens
    Output: (batch, seq_len, vocab_size) - logits
    """
    
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=128,
                 num_layers=2, max_len=64, dropout=0.0):
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
        
        # LM Head
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
        
        # Embeddings (gather diferenciable)
        tok_emb = self.token_emb.gather(input_data.reshape(-1))  # (batch*seq, d_model)
        tok_emb = tok_emb.reshape(batch, seq_len, self.d_model)
        
        # Positional embeddings (gather)
        pos_indices = np.arange(seq_len)
        pos_emb = self.pos_emb.gather(pos_indices)  # (seq_len, d_model)
        pos_emb = pos_emb.reshape(1, seq_len, self.d_model)
        
        # Suma: broadcast de pos_emb sobre batch
        h = tok_emb + pos_emb
        
        # Bloques
        for block in self.blocks:
            h = block.forward(h, causal_mask=True)
        
        # Final LayerNorm
        h = self.ln_final.forward(h)
        
        # LM Head: logits = h @ W_lm_head
        logits = h.batch_matmul(self.lm_head)
        
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """Genera tokens a partir de un prompt"""
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = ids[-self.max_len:]
            x = Tensor(np.array([context]))
            logits = self.forward(x).data
            
            last_logits = logits[0, -1] / temperature
            
            if top_k is not None:
                top_indices = np.argsort(last_logits)[-top_k:]
                mask = np.full_like(last_logits, -1e9)
                mask[top_indices] = last_logits[top_indices]
                last_logits = mask
            
            exp_l = np.exp(last_logits - np.max(last_logits))
            probs = exp_l / np.sum(exp_l)
            
            next_id = int(np.random.choice(self.vocab_size, p=probs))
            ids.append(next_id)
        
        return ids
    
    def summary(self):
        super().summary()
        print(f"   Vocab: {self.vocab_size}, d_model: {self.d_model}")
        print(f"   Heads: {self.num_heads}, Layers: {self.num_layers}")


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO TRANSFORMER V3.0 (backprop real)")
    print("="*60)
    
    model = Transformer(vocab_size=20, d_model=16, num_heads=2,
                        d_ff=32, num_layers=1, max_len=8)
    model.summary()
    
    # Forward
    x = Tensor(np.array([[1, 2, 3, 4]]))
    logits = model.forward(x)
    print(f"\n📊 Input: {x.data.shape}")
    print(f"📊 Logits: {logits.data.shape}")
    
    # Backward
    loss = logits.sum()
    loss.backward()
    
    # Verificar que TODOS los parámetros tienen gradiente
    print(f"\n🔍 Verificando gradientes:")
    all_have_grad = True
    for i, param in enumerate(model.parameters()):
        has = param.grad is not None
        if not has:
            print(f"   ❌ Parámetro {i} sin gradiente: shape={param.data.shape}")
            all_have_grad = False
    
    if all_have_grad:
        print(f"   ✅ TODOS los {len(model.parameters())} parámetros tienen gradiente")
    
    print("\n✅ TRANSFORMER V3.0 FUNCIONANDO CON BACKPROP REAL")
