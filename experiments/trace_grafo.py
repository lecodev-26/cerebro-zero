"""
Tracear el grafo de autograd paso a paso.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor


def trace():
    np.random.seed(42)
    
    vocab_size = 10
    d_model = 8
    seq_len = 5
    
    lim = 1.0 / np.sqrt(d_model)
    token_emb = Tensor(np.random.uniform(-lim, lim, (vocab_size, d_model)), requires_grad=True)
    lm_head = Tensor(np.random.uniform(-lim, lim, (d_model, vocab_size)), requires_grad=True)
    
    X = np.array([[1, 2, 3, 4, 5]])
    y = X.copy()
    
    print("🔍 TRACE DEL GRAFO")
    print("="*60)
    
    print(f"0. token_emb: requires_grad={token_emb.requires_grad}")
    print(f"   lm_head:   requires_grad={lm_head.requires_grad}")
    
    # 1. Gather
    indices = X.reshape(-1).astype(int)
    h1 = token_emb.gather(indices)
    print(f"\n1. gather: shape={h1.data.shape}, requires_grad={h1.requires_grad}")
    print(f"   h1._prev: {[p._op or 'input' for p in h1._prev]}")
    
    # 2. Reshape
    h2 = h1.reshape(1, seq_len, d_model)
    print(f"\n2. reshape: shape={h2.data.shape}, requires_grad={h2.requires_grad}")
    print(f"   h2._prev: {[p._op or 'input' for p in h2._prev]}")
    
    # 3. Batch matmul
    logits = h2.batch_matmul(lm_head)
    print(f"\n3. batch_matmul: shape={logits.data.shape}, requires_grad={logits.requires_grad}")
    print(f"   logits._prev: {[p._op or 'input' for p in logits._prev]}")
    
    # 4. Softmax
    probs = logits.softmax(axis=-1)
    print(f"\n4. softmax: shape={probs.data.shape}, requires_grad={probs.requires_grad}")
    print(f"   probs._prev: {[p._op or 'input' for p in probs._prev]}")
    
    # 5. Add eps (probs + 1e-10)
    # OJO: np.float32 + float puede dar problemas
    eps = 1e-10
    probs_clipped = probs + eps
    print(f"\n5. +eps: shape={probs_clipped.data.shape}, requires_grad={probs_clipped.requires_grad}")
    print(f"   probs_clipped._prev: {[p._op or 'input' for p in probs_clipped._prev]}")
    
    # 6. Log
    log_probs = probs_clipped.log()
    print(f"\n6. log: shape={log_probs.data.shape}, requires_grad={log_probs.requires_grad}")
    print(f"   log_probs._prev: {[p._op or 'input' for p in log_probs._prev]}")
    
    # 7. Multiplicar por one-hot
    targets_onehot = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
    for b in range(1):
        for s in range(seq_len):
            targets_onehot[b, s, y[b, s]] = 1.0
    
    onehot_tensor = Tensor(targets_onehot, requires_grad=False)
    print(f"\n7. onehot: shape={onehot_tensor.data.shape}, requires_grad={onehot_tensor.requires_grad}")
    
    product = log_probs * onehot_tensor
    print(f"\n8. product: shape={product.data.shape}, requires_grad={product.requires_grad}")
    print(f"   product._prev: {[p._op or 'input' for p in product._prev]}")
    
    # 9. Sum
    summed = product.sum(axis=-1)
    print(f"\n9. sum(axis=-1): shape={summed.data.shape}, requires_grad={summed.requires_grad}")
    print(f"   summed._prev: {[p._op or 'input' for p in summed._prev]}")
    
    # 10. Mean
    loss = summed.mean()
    print(f"\n10. mean: shape={loss.data.shape}, requires_grad={loss.requires_grad}")
    print(f"    loss._prev: {[p._op or 'input' for p in loss._prev]}")
    
    # 11. Negativo
    loss_final = -loss
    print(f"\n11. -loss: shape={loss_final.data.shape}, requires_grad={loss_final.requires_grad}")
    print(f"    loss_final._prev: {[p._op or 'input' for p in loss_final._prev]}")
    print(f"    loss_final._op: {loss_final._op}")
    
    # 12. Backward
    print("\n" + "="*60)
    print("🔙 BACKWARD")
    print("="*60)
    
    loss_final.backward()
    
    print(f"Después del backward:")
    print(f"  token_emb.grad: {'None' if token_emb.grad is None else f'max={np.max(np.abs(token_emb.grad)):.6f}'}")
    print(f"  lm_head.grad:   {'None' if lm_head.grad is None else f'max={np.max(np.abs(lm_head.grad)):.6f}'}")
    print(f"  logits.grad:    {'None' if logits.grad is None else f'max={np.max(np.abs(logits.grad)):.6f}'}")
    print(f"  probs.grad:     {'None' if probs.grad is None else f'max={np.max(np.abs(probs.grad)):.6f}'}")
    print(f"  log_probs.grad: {'None' if log_probs.grad is None else f'max={np.max(np.abs(log_probs.grad)):.6f}'}")
    print(f"  product.grad:   {'None' if product.grad is None else f'max={np.max(np.abs(product.grad)):.6f}'}")
    print(f"  summed.grad:    {'None' if summed.grad is None else f'max={np.max(np.abs(summed.grad)):.6f}'}")
    
    # Verificar topología del backward
    print("\n🔍 TOPOLOGÍA DEL GRAFO:")
    print(f"  loss_final._prev: {[(p._op, id(p)) for p in loss_final._prev]}")
    print(f"  loss._prev:       {[(p._op, id(p)) for p in loss._prev]}")
    print(f"  summed._prev:     {[(p._op, id(p)) for p in summed._prev]}")
    print(f"  product._prev:    {[(p._op, id(p)) for p in product._prev]}")
    print(f"  log_probs._prev:  {[(p._op, id(p)) for p in log_probs._prev]}")
    print(f"  probs_clipped._prev: {[(p._op, id(p)) for p in probs_clipped._prev]}")
    print(f"  probs._prev:      {[(p._op, id(p)) for p in probs._prev]}")
    print(f"  logits._prev:     {[(p._op, id(p)) for p in logits._prev]}")
    print(f"  h2._prev:         {[(p._op, id(p)) for p in h2._prev]}")
    print(f"  h1._prev:         {[(p._op, id(p)) for p in h1._prev]}")


if __name__ == "__main__":
    trace()
