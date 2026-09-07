import numpy as np

class Tensor:
    """Tensor propio para control total de operaciones"""
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
    
    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __mul__(self, other):
        return Tensor(self.data * other.data)
    
    def matmul(self, other):
        return Tensor(np.dot(self.data, other.data))
    
    def relu(self):
        return Tensor(np.maximum(0, self.data))
    
    def softmax(self):
        exp = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        return Tensor(exp / np.sum(exp, axis=-1, keepdims=True))
    
    def reshape(self, shape):
        return Tensor(self.data.reshape(shape))
    
    def sum(self, axis=None):
        return Tensor(np.sum(self.data, axis=axis))
    
    def mean(self, axis=None):
        return Tensor(np.mean(self.data, axis=axis))
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, grad={self.grad is not None})"
