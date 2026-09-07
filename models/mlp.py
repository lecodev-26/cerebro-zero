import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.layers import Linear
from core.activations import ReLU, Sigmoid, Softmax

class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations = []
        
        for i in range(len(layer_sizes)-1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.activations.append(ReLU())
            else:
                self.activations.append(Sigmoid())
    
    def forward(self, x, return_all=False):
        self.cache = [x]
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            z = layer.forward(x)
            if i == len(self.layers) - 1:
                a = act.forward(z)
            else:
                a = act.forward(z)
            self.cache.append(a)
            x = a
        return x
    
    def backward(self, grad):
        for i in range(len(self.layers)-1, -1, -1):
            if i < len(self.layers) - 1:
                grad = self.activations[i].backward(self.cache[i+1], grad)
            grad = self.layers[i].backward(grad)
        return grad
    
    def train_step(self, x, y, optimizer):
        output = self.forward(x)
        loss = np.mean((output - y) ** 2)
        grad = 2 * (output - y) / len(y)
        self.backward(grad)
        optimizer.step()
        return loss
    
    def predict(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"MLP({[str(l) for l in self.layers]})"
