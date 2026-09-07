import numpy as np

class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr
    
    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(self.lr)

class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        
        for layer in layers:
            if hasattr(layer, 'weight'):
                self.m_w.append(np.zeros_like(layer.weight))
                self.v_w.append(np.zeros_like(layer.weight))
                self.m_b.append(np.zeros_like(layer.bias))
                self.v_b.append(np.zeros_like(layer.bias))
    
    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.grad_weight
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (layer.grad_weight ** 2)
                m_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                layer.weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_bias
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_bias ** 2)
                m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
                layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
