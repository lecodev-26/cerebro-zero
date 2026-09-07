import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.module import Module
from core.tensor_v2 import Tensor, zeros, randn

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        lim = np.sqrt(6.0 / (in_features + out_features))
        weight = Tensor(np.random.uniform(-lim, lim, (in_features, out_features)), requires_grad=True)
        self.weight = weight
        self._params.append(weight)
        
        if bias:
            bias_t = zeros((1, out_features))
            bias_t.requires_grad = True
            self.bias = bias_t
            self._params.append(bias_t)
        else:
            self.bias = None
    
    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def __repr__(self):
        return f"Linear({self.in_features} → {self.out_features})"

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            self.layers.append(layer)
            self._children.append(layer)
            setattr(self, f"layer_{i}", layer)
    
    def forward(self, x):
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
    def forward(self, x):
        return x.reshape(x.data.shape[0], -1)
    
    def __repr__(self):
        return "Flatten()"
