"""
Optimizadores: SGD, SGD con Momentum y Adam
"""

import numpy as np

class SGD:
    """Stochastic Gradient Descent con opción de Momentum"""
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters if p.requires_grad]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                # Momentum
                if self.momentum > 0:
                    self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad
                    param.data += self.velocities[i]
                else:
                    param.data -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = None

class Adam:
    """Adam Optimizer (Kingma & Ba, 2014)"""
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Momentos
        self.m = [np.zeros_like(p.data) for p in self.parameters if p.requires_grad]
        self.v = [np.zeros_like(p.data) for p in self.parameters if p.requires_grad]
    
    def step(self):
        self.t += 1
        grad_idx = 0
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                # Actualizar momentos
                self.m[grad_idx] = self.beta1 * self.m[grad_idx] + (1 - self.beta1) * param.grad
                self.v[grad_idx] = self.beta2 * self.v[grad_idx] + (1 - self.beta2) * (param.grad ** 2)
                
                # Corrección de sesgo
                m_hat = self.m[grad_idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[grad_idx] / (1 - self.beta2 ** self.t)
                
                # Actualizar parámetros
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                grad_idx += 1
    
    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = None
