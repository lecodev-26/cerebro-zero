import numpy as np

class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x, grad):
        return grad * (x > 0).astype(float)

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def backward(x, grad):
        sig = Sigmoid.forward(x)
        return grad * sig * (1 - sig)

class Softmax:
    @staticmethod
    def forward(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def backward(x, grad):
        return grad  # Simplificado para cross-entropy

class Tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(x, grad):
        return grad * (1 - np.tanh(x) ** 2)
