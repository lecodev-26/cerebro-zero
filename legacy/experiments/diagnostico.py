"""
Diagnóstico incremental del Transformer
Empezamos simple y añadimos complejidad.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor
from core.optimizers import Adam
from core.layernorm import LayerNorm


# ============================================
# TEST 1: EMBEDDING + LM HEAD SOLO
# ============================================

class TinyModel1:
    """Solo embedding + LM head. Sin attention, sin FFN."""
    def __init__(self, vocab_size, d_model):
        lim = 1.0 / np.sqrt(d_model)
        self.token_emb = Tensor(np.random.uniform(-lim, lim, (vocab_size, d_model)), requires_grad=True)
        self.lm_head = Tensor(np.random.uniform(-lim, lim, (d_model, vocab_size)), requires_grad=True)
    
    def parameters(self):
        return [self.token_emb, self.lm_head]
    
    def forward(self, x):
        # x: (batch, seq_len) - índices
        if not isinstance(x, Tensor):
            x = Tensor(x)
        input_data = x.data.astype(int)
        batch, seq_len = input_data.shape
        
        # Embedding via gather
        h = self.token_emb.gather(input_data.reshape(-1))
        h = h.reshape(batch, seq_len, self.token_emb.data.shape[1])
        
        # LM Head
        logits = h.batch_matmul(self.lm_head)
        return logits


# ============================================
# TEST 2: + POSITIONAL EMBEDDING
# ============================================

class TinyModel2:
    def __init__(self, vocab_size, d_model, max_len):
        lim = 1.0 / np.sqrt(d_model)
        self.token_emb = Tensor(np.random.uniform(-lim, lim, (vocab_size, d_model)), requires_grad=True)
        self.pos_emb = Tensor(np.random.uniform(-lim, lim, (max_len, d_model)), requires_grad=True)
        self.lm_head = Tensor(np.random.uniform(-lim, lim, (d_model, vocab_size)), requires_grad=True)
        self.max_len = max_len
    
    def parameters(self):
        return [self.token_emb, self.pos_emb, self.lm_head]
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        input_data = x.data.astype(int)
        batch, seq_len = input_data.shape
        seq_len = min(seq_len, self.max_len)
        input_data = input_data[:, :seq_len]
        
        h = self.token_emb.gather(input_data.reshape(-1))
        h = h.reshape(batch, seq_len, self.token_emb.data.shape[1])
        
        pos = self.pos_emb.gather(np.arange(seq_len))
        pos = pos.reshape(1, seq_len, self.pos_emb.data.shape[1])
        h = h + pos
        
        logits = h.batch_matmul(self.lm_head)
        return logits


# ============================================
# TEST 3: + LAYERNORM
# ============================================

class TinyModel3:
    def __init__(self, vocab_size, d_model, max_len):
        lim = 1.0 / np.sqrt(d_model)
        self.token_emb = Tensor(np.random.uniform(-lim, lim, (vocab_size, d_model)), requires_grad=True)
        self.pos_emb = Tensor(np.random.uniform(-lim, lim, (max_len, d_model)), requires_grad=True)
        self.ln = LayerNorm(d_model)
        self.lm_head = Tensor(np.random.uniform(-lim, lim, (d_model, vocab_size)), requires_grad=True)
        self.max_len = max_len
    
    def parameters(self):
        params = [self.token_emb, self.pos_emb, self.lm_head]
        params.extend(self.ln.parameters())
        return params
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        input_data = x.data.astype(int)
        batch, seq_len = input_data.shape
        seq_len = min(seq_len, self.max_len)
        input_data = input_data[:, :seq_len]
        
        h = self.token_emb.gather(input_data.reshape(-1))
        h = h.reshape(batch, seq_len, self.token_emb.data.shape[1])
        
        pos = self.pos_emb.gather(np.arange(seq_len))
        pos = pos.reshape(1, seq_len, self.pos_emb.data.shape[1])
        h = h + pos
        
        h = self.ln.forward(h)
        
        logits = h.batch_matmul(self.lm_head)
        return logits


# ============================================
# CROSS-ENTROPY
# ============================================

def cross_entropy(logits, targets):
    batch, seq_len, vocab_size = logits.data.shape
    probs = logits.softmax(axis=-1)
    eps = 1e-10
    probs_clipped = probs + eps
    log_probs = probs_clipped.log()
    
    targets_onehot = np.zeros((batch, seq_len, vocab_size), dtype=log_probs.data.dtype)
    for b in range(batch):
        for s in range(seq_len):
            targets_onehot[b, s, targets[b, s]] = 1.0
    
    product = log_probs * Tensor(targets_onehot, requires_grad=False)
    summed = product.sum(axis=-1)
    return -summed.mean()


# ============================================
# ENTRENAR
# ============================================

def train_model(model, X_data, y_data, lr=0.1, epochs=100, batch_size=16, name="Model"):
    optimizer = Adam(model.parameters(), lr=lr)
    
    print(f"\n🏋️ Entrenando {name}...")
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(5):  # 5 batches por época
            idx = np.random.choice(len(X_data), batch_size, replace=False)
            X_batch = X_data[idx]
            y_batch = y_data[idx]
            
            # Forward
            logits = model.forward(Tensor(X_batch))
            loss = cross_entropy(logits, y_batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_losses.append(float(loss.data))
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}: loss = {avg_loss:.4f}")
    
    return losses


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("🧪 DIAGNÓSTICO INCREMENTAL")
    print("="*60)
    
    # Dataset simple: copiar
    vocab_size = 10
    seq_len = 5
    num_samples = 200
    
    X_data = np.random.randint(1, vocab_size, (num_samples, seq_len))
    y_data = X_data.copy()
    
    print(f"📊 Dataset: copiar secuencia de {seq_len} tokens, vocab={vocab_size}")
    print(f"   Loss inicial esperado (random): ln({vocab_size}) = {np.log(vocab_size):.4f}")
    
    # Test 1: Embedding + LM Head
    print("\n" + "="*60)
    print("TEST 1: Embedding + LM Head (sin pos, sin LN)")
    print("="*60)
    model1 = TinyModel1(vocab_size, d_model=32)
    losses1 = train_model(model1, X_data, y_data, lr=0.1, epochs=100, name="Tiny1")
    print(f"   Mejora: {(losses1[0] - losses1[-1])/losses1[0]*100:.1f}%")
    
    # Test 2: + Posicional
    print("\n" + "="*60)
    print("TEST 2: + Positional Embedding")
    print("="*60)
    model2 = TinyModel2(vocab_size, d_model=32, max_len=seq_len)
    losses2 = train_model(model2, X_data, y_data, lr=0.1, epochs=100, name="Tiny2")
    print(f"   Mejora: {(losses2[0] - losses2[-1])/losses2[0]*100:.1f}%")
    
    # Test 3: + LayerNorm
    print("\n" + "="*60)
    print("TEST 3: + LayerNorm")
    print("="*60)
    model3 = TinyModel3(vocab_size, d_model=32, max_len=seq_len)
    losses3 = train_model(model3, X_data, y_data, lr=0.1, epochs=100, name="Tiny3")
    print(f"   Mejora: {(losses3[0] - losses3[-1])/losses3[0]*100:.1f}%")
    
    print("\n" + "="*60)
    print("📊 RESUMEN:")
    print(f"   Test 1 (Emb+LM):     {losses1[0]:.4f} → {losses1[-1]:.4f} ({((losses1[0]-losses1[-1])/losses1[0]*100):+.1f}%)")
    print(f"   Test 2 (+Pos):       {losses2[0]:.4f} → {losses2[-1]:.4f} ({((losses2[0]-losses2[-1])/losses2[0]*100):+.1f}%)")
    print(f"   Test 3 (+LN):        {losses3[0]:.4f} → {losses3[-1]:.4f} ({((losses3[0]-losses3[-1])/losses3[0]*100):+.1f}%)")
    print("="*60)
