"""
Tensor V2 - Autograd simplificado y robusto
"""

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
    
    # Operaciones básicas
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._backward = self._make_backward_add(other, out)
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._backward = self._make_backward_mul(other, out)
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._backward = self._make_backward_sub(other, out)
        return out
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._backward = self._make_backward_matmul(other, out)
        return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_reshape(out)
        return out
    
    def transpose(self):
        out = Tensor(self.data.T,
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_transpose(out)
        return out
    
    def sum(self, axis=None):
        out = Tensor(np.sum(self.data, axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_sum(axis, out)
        return out
    
    def mean(self, axis=None):
        out = Tensor(np.mean(self.data, axis=axis),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_mean(axis, out)
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_exp(out)
        return out
    
    def log(self):
        out = Tensor(np.log(np.abs(self.data) + 1e-10),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_log(out)
        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(np.abs(self.data) + 1e-10),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._backward = self._make_backward_sqrt(out)
        return out
    
    # === BACKWARD FUNCTIONS ===
    
    def _make_backward_add(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = (out.grad if self.grad is None else self.grad + out.grad)
            if other.requires_grad:
                other.grad = (out.grad if other.grad is None else other.grad + out.grad)
        return backward
    
    def _make_backward_mul(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = (other.data * out.grad if self.grad is None else self.grad + other.data * out.grad)
            if other.requires_grad:
                other.grad = (self.data * out.grad if other.grad is None else other.grad + self.data * out.grad)
        return backward
    
    def _make_backward_sub(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad = (out.grad if self.grad is None else self.grad + out.grad)
            if other.requires_grad:
                other.grad = (-out.grad if other.grad is None else other.grad - out.grad)
        return backward
    
    def _make_backward_matmul(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = np.dot(out.grad, other.data.T)
                self.grad = (grad if self.grad is None else self.grad + grad)
            if other.requires_grad:
                grad = np.dot(self.data.T, out.grad)
                other.grad = (grad if other.grad is None else other.grad + grad)
        return backward
    
    def _make_backward_reshape(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def _make_backward_transpose(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad.T
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def _make_backward_sum(self, axis, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if axis is None:
                    grad = np.full_like(self.data, out.grad)
                else:
                    grad = np.expand_dims(out.grad, axis)
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def _make_backward_mean(self, axis, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if axis is None:
                    grad = np.full_like(self.data, out.grad / self.data.size)
                else:
                    grad = np.expand_dims(out.grad / self.data.shape[axis], axis)
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def _make_backward_exp(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad * np.exp(self.data)
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def _make_backward_log(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad / self.data
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def _make_backward_sqrt(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad * 0.5 / np.sqrt(self.data + 1e-10)
                self.grad = (grad if self.grad is None else self.grad + grad)
        return backward
    
    def backward(self):
        self.grad = np.ones_like(self.data)
        self._propagar()
    
    def _propagar(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build(prev)
                topo.append(v)
        build(self)
        for v in reversed(topo):
            if v.requires_grad:
                v._backward()
    
    def zero_grad(self):
        self.grad = None
        for prev in self._prev:
            if hasattr(prev, 'zero_grad'):
                prev.zero_grad()
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, grad={self.grad is not None})"

# Funciones de creación
def tensor(data):
    return Tensor(data)

def ones(shape):
    return Tensor(np.ones(shape))

def zeros(shape):
    return Tensor(np.zeros(shape))

def randn(shape):
    return Tensor(np.random.randn(*shape))

def rand(shape):
    return Tensor(np.random.rand(*shape))

def arange(start, stop, step=1):
    return Tensor(np.arange(start, stop, step))

def eye(n):
    return Tensor(np.eye(n))
