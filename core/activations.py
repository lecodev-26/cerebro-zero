import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.module import Module
from core.tensor_v2 import Tensor

class ReLU(Module):
    def forward(self, x):
        data = x.data
        out_data = np.maximum(0, data)
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,))
        
        def backward():
            if x.requires_grad and out.grad is not None:
                grad = out.grad * (data > 0).astype(float)
                x.grad = grad if x.grad is None else x.grad + grad
        
        out._backward = backward
        return out
    
    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    def forward(self, x):
        data = x.data
        out_data = 1 / (1 + np.exp(-np.clip(data, -500, 500)))
        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,))
        
        def backward():
            if x.requires_grad and out.grad is not None:
                grad = out.grad * out_data * (1 - out_data)
                x.grad = grad if x.grad is None else x.grad + grad
        
        out._backward = backward
        return out
    
    def __repr__(self):
        return "Sigmoid()"
