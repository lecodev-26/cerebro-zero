"""
Funciones de activación - Compatibles con Tensor V3.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.module import Module
from core.tensor import Tensor


class ReLU(Module):
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x.sigmoid()
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x.tanh()
    
    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x.softmax(axis=self.dim)
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"
