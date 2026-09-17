"""
Capas básicas para redes neuronales - Corregido para Tensor V3.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.module import Module
from core.tensor import Tensor, zeros, randn


class Linear(Module):
    """Capa lineal: y = x @ W + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicialización Xavier
        lim = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-lim, lim, (in_features, out_features))
        self.weight = Tensor(weight_data, requires_grad=True)
        self._params.append(self.weight)
        
        if bias:
            bias_data = np.zeros((1, out_features))
            self.bias = Tensor(bias_data, requires_grad=True)
            self._params.append(self.bias)
        else:
            self.bias = None
    
    def forward(self, x):
        """
        x: Tensor con shape (batch, in_features)
        """
        # Asegurar que x es Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def __repr__(self):
        return f"Linear({self.in_features} → {self.out_features})"


class Sequential(Module):
    """Contenedor de capas secuenciales"""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            self.layers.append(layer)
            self._children.append(layer)
            setattr(self, f"layer_{i}", layer)
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        out = "Sequential(\n"
        for i, layer in enumerate(self.layers):
            out += f"  ({i}): {repr(layer)}\n"
        out += ")"
        return out


class Flatten(Module):
    """Aplana la entrada"""
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        batch_size = x.data.shape[0]
        return x.reshape(batch_size, -1)
    
    def __repr__(self):
        return "Flatten()"
