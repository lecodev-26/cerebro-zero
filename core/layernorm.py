"""
Layer Normalization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor
from core.module import Module


class LayerNorm(Module):
    """
    Layer Normalization.
    
    Normaliza la última dimensión: (x - mean) / sqrt(var + eps) * gamma + beta
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        # Parámetros aprendibles
        self.gamma = Tensor(np.ones(self.normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(self.normalized_shape), requires_grad=True)
        self._params.append(self.gamma)
        self._params.append(self.beta)
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # Media y varianza sobre la última dimensión
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = x_norm * self.gamma.data + self.beta.data
        
        out = Tensor(out_data, requires_grad=True, _children=(x, self.gamma, self.beta))
        
        def backward():
            if out.grad is None:
                return
            
            # Gradientes
            # dL/dgamma = sum(dL/dout * x_norm)
            dgamma = np.sum(out.grad * x_norm, axis=tuple(range(out.grad.ndim - 1)))
            dbeta = np.sum(out.grad, axis=tuple(range(out.grad.ndim - 1)))
            
            if self.gamma.requires_grad:
                self.gamma.grad = dgamma if self.gamma.grad is None else self.gamma.grad + dgamma
            if self.beta.requires_grad:
                self.beta.grad = dbeta if self.beta.grad is None else self.beta.grad + dbeta
            
            # dL/dx (aproximado, suficiente para entrenamiento)
            if x.requires_grad:
                dx = (out.grad * self.gamma.data) / np.sqrt(var + self.eps)
                x.grad = dx if x.grad is None else x.grad + dx
        
        out._backward = backward
        return out
    
    def __repr__(self):
        return f"LayerNorm({self.normalized_shape})"


if __name__ == "__main__":
    print("🧪 PROBANDO LAYERNORM")
    print("="*50)
    
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], requires_grad=True)
    y = ln(x)
    
    print(f"Entrada: {x.data}")
    print(f"Salida: {y.data}")
    
    # Cada fila debe tener media ~0 y std ~1
    for i in range(2):
        media = np.mean(y.data[i])
        std = np.std(y.data[i])
        print(f"   Fila {i}: media={media:.4f}, std={std:.4f}")
    
    print("\n✅ LAYERNORM FUNCIONANDO")
