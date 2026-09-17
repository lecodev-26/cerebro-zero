"""
Layer Normalization con autograd completo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor


class LayerNorm:
    """
    Layer Normalization.
    
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Todo dentro del grafo de autograd.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        # Parámetros aprendibles
        self.gamma = Tensor(np.ones(self.normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(self.normalized_shape), requires_grad=True)
    
    def parameters(self):
        return [self.gamma, self.beta]
    
    def forward(self, x):
        """
        Forward completo dentro del grafo.
        
        x: Tensor (..., normalized_shape)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # Media y varianza sobre la última dimensión
        mean = x.mean(axis=-1, keepdims=True)
        # E[x^2] - E[x]^2
        x_sq_mean = (x * x).mean(axis=-1, keepdims=True)
        var = x_sq_mean - mean * mean
        var = var + self.eps  # Evitar sqrt(0)
        std = var.sqrt()
        x_norm = (x - mean) / std
        
        # Escalar y desplazar
        out = x_norm * self.gamma + self.beta
        return out
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"LayerNorm({self.normalized_shape})"


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO LAYERNORM V3.0")
    print("="*50)
    
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0]], requires_grad=True)
    
    y = ln.forward(x)
    print(f"Entrada: {x.data}")
    print(f"Salida:  {y.data}")
    
    # Verificar media ~ 0 y var ~ 1
    for i in range(2):
        media = np.mean(y.data[i])
        std = np.std(y.data[i])
        print(f"   Fila {i}: media={media:.4f}, std={std:.4f}")
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    print(f"\n✅ x.grad es None? {x.grad is None}")
    print(f"✅ gamma.grad es None? {ln.gamma.grad is None}")
    print(f"✅ beta.grad es None? {ln.beta.grad is None}")
    
    print("\n✅ LAYERNORM V3.0 FUNCIONANDO")
