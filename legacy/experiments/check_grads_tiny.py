"""
Verificar que los gradientes de Tiny1 son correctos.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor


def check_gradients():
    np.random.seed(42)
    
    vocab_size = 10
    d_model = 8
    seq_len = 5
    
    # Modelo
    lim = 1.0 / np.sqrt(d_model)
    token_emb = Tensor(np.random.uniform(-lim, lim, (vocab_size, d_model)), requires_grad=True)
    lm_head = Tensor(np.random.uniform(-lim, lim, (d_model, vocab_size)), requires_grad=True)
    
    # Input
    X = np.array([[1, 2, 3, 4, 5]])
    y = X.copy()
    
    print("🧪 Forward + Backward en Tiny1")
    print("="*60)
    
    # Forward
    input_data = X.astype(int)
    batch, seq_len = input_data.shape
    
    h = token_emb.gather(input_data.reshape(-1))
    print(f"1. Tras gather: shape={h.data.shape}, requires_grad={h.requires_grad}")
    
    h = h.reshape(batch, seq_len, d_model)
    print(f"2. Tras reshape: shape={h.data.shape}, requires_grad={h.requires_grad}")
    
    logits = h.batch_matmul(lm_head)
    print(f"3. Tras batch_matmul: shape={logits.data.shape}, requires_grad={logits.requires_grad}")
    
    # Cross-entropy loss
    probs = logits.softmax(axis=-1)
    eps = 1e-10
    probs_clipped = probs + eps
    log_probs = probs_clipped.log()
    
    targets_onehot = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            targets_onehot[b, s, y[b, s]] = 1.0
    
    product = log_probs * Tensor(targets_onehot, requires_grad=False)
    summed = product.sum(axis=-1)
    loss = -summed.mean()
    
    print(f"4. Loss: {loss.data:.6f}")
    
    # Backward
    loss.backward()
    print()
    print(f"5. Gradientes después del backward:")
    print(f"   token_emb.grad:  {'None' if token_emb.grad is None else f'shape={token_emb.grad.shape}, max={np.max(np.abs(token_emb.grad)):.6f}'}")
    print(f"   lm_head.grad:    {'None' if lm_head.grad is None else f'shape={lm_head.grad.shape}, max={np.max(np.abs(lm_head.grad)):.6f}'}")
    
    # Gradiente numérico para lm_head
    print()
    print("6. Comparando gradiente analítico vs numérico:")
    
    def loss_fn_lm(w_data):
        lm_head.data = w_data.copy()
        h = token_emb.gather(input_data.reshape(-1))
        h = h.reshape(batch, seq_len, d_model)
        logits = h.batch_matmul(lm_head)
        probs = logits.softmax(axis=-1)
        probs_clipped = probs + eps
        log_probs = probs_clipped.log()
        product = log_probs * Tensor(targets_onehot, requires_grad=False)
        summed = product.sum(axis=-1)
        return float(-summed.mean().data)
    
    # Verificar 3 elementos
    lm_data = lm_head.data.copy()
    analytic = lm_head.grad.copy()
    eps_fd = 1e-4
    
    for i in range(3):
        idx = np.random.randint(0, lm_data.size)
        multi = np.unravel_index(idx, lm_data.shape)
        
        old = lm_data[multi]
        
        lm_data[multi] = old + eps_fd
        f_plus = loss_fn_lm(lm_data)
        
        lm_data[multi] = old - eps_fd
        f_minus = loss_fn_lm(lm_data)
        
        lm_data[multi] = old
        lm_head.data = lm_data.copy()
        
        numeric = (f_plus - f_minus) / (2 * eps_fd)
        analytic_val = analytic.flatten()[idx]
        
        match = "✅" if abs(numeric - analytic_val) < 1e-4 else "❌"
        print(f"   {match} idx {multi}: analytic={analytic_val:.6f}, numeric={numeric:.6f}, diff={abs(numeric - analytic_val):.6f}")


if __name__ == "__main__":
    check_gradients()
