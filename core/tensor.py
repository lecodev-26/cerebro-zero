"""
Tensor V3.2 - Con soporte float32/float64 y numérica estable
"""

import numpy as np

# Dtype global (float64 por defecto, float32 para móvil)
DEFAULT_DTYPE = np.float64


def set_dtype(dtype):
    """Configura el dtype por defecto (float32 o float64)"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype
    print(f"🔧 Dtype global: {dtype}")


def get_dtype():
    return DEFAULT_DTYPE


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        
        target_dtype = dtype if dtype is not None else DEFAULT_DTYPE
        self.data = np.array(data, dtype=target_dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = ''
    
    # ============================================
    # OPERACIONES BÁSICAS
    # ============================================
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = '+'
        out._backward = self._make_backward_add(other, out)
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = '*'
        out._backward = self._make_backward_mul(other, out)
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = '-'
        out._backward = self._make_backward_sub(other, out)
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = '/'
        out._backward = self._make_backward_div(other, out)
        return out
    
    def __pow__(self, n):
        return self.pow(n)
    
    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,))
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = 'matmul'
        out._backward = self._make_backward_matmul(other, out)
        return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._op = 'reshape'
        out._backward = self._make_backward_reshape(out)
        return out
    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes),
                     requires_grad=self.requires_grad,
                     _children=(self,))
        out._op = 'transpose'
        out._backward = self._make_backward_transpose(out, axes)
        return out
    
    @property
    def T(self):
        return self.transpose()
    
    # ============================================
    # REDUCCIONES
    # ============================================
    
    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'sum'
        out._backward = self._make_backward_sum(axis, keepdims, out)
        return out
    
    def mean(self, axis=None, keepdims=False):
        # Verificar que no está vacío
        if self.data.size == 0:
            raise ValueError("mean() no funciona en tensor vacío")
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'mean'
        out._backward = self._make_backward_mean(axis, keepdims, out)
        return out
    
    # ============================================
    # FUNCIONES MATEMÁTICAS
    # ============================================
    
    def exp(self):
        out_data = np.exp(np.clip(self.data, -700, 700))
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'exp'
        out._backward = self._make_backward_exp(out)
        return out
    
    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("log() requiere x > 0")
        out_data = np.log(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'log'
        out._backward = self._make_backward_log(out)
        return out
    
    def sqrt(self):
        if np.any(self.data < 0):
            raise ValueError("sqrt() requiere x >= 0")
        out_data = np.sqrt(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'sqrt'
        out._backward = self._make_backward_sqrt(out)
        return out
    
    def pow(self, n):
        out_data = self.data ** n
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = f'pow({n})'
        out._backward = self._make_backward_pow(n, out)
        return out
    
    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'relu'
        out._backward = self._make_backward_relu(out)
        return out
    
    def sigmoid(self):
        """Sigmoid numéricamente estable (evita overflow)"""
        data = self.data
        # Clipear para evitar overflow
        data_clipped = np.clip(data, -500, 500)
        out_data = np.where(
            data_clipped >= 0,
            1 / (1 + np.exp(-data_clipped)),
            np.exp(data_clipped) / (1 + np.exp(data_clipped))
        )
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'sigmoid'
        out._backward = self._make_backward_sigmoid(out_data, out)
        return out
    
    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'tanh'
        out._backward = self._make_backward_tanh(out_data, out)
        return out
    
    def softmax(self, axis=-1):
        """Softmax numéricamente estable (resta el máximo)"""
        data = self.data
        data_max = np.max(data, axis=axis, keepdims=True)
        data_shifted = data - data_max
        exp_data = np.exp(data_shifted)
        out_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'softmax'
        out._backward = self._make_backward_softmax(out_data, axis, out)
        return out
    
    # ============================================
    # HELPER: REDUCIR GRADIENTE (broadcasting)
    # ============================================
    
    def _reduce_grad(self, grad, original_shape):
        """Reduce un gradiente a la forma original después de broadcasting"""
        if grad.shape == original_shape:
            return grad
        
        # Sumar dimensiones extra añadidas al principio
        while len(grad.shape) > len(original_shape):
            grad = np.sum(grad, axis=0)
        
        # Sumar dimensiones que eran 1
        for i, dim in enumerate(original_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        
        return grad.reshape(original_shape)
    
    # ============================================
    # BACKWARD FUNCTIONS
    # ============================================
    
    def _make_backward_add(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = self._reduce_grad(out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = other._reduce_grad(out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_mul(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = self._reduce_grad(other.data * out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = other._reduce_grad(self.data * out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_sub(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = self._reduce_grad(out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = other._reduce_grad(-out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_div(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = self._reduce_grad(out.grad / other.data, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = other._reduce_grad(-out.grad * self.data / (other.data ** 2), other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_matmul(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = np.dot(out.grad, other.data.T)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = np.dot(self.data.T, out.grad)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_reshape(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad.reshape(self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_transpose(self, out, axes):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if axes:
                    inv_axes = np.argsort(axes)
                    g = out.grad.transpose(*inv_axes)
                else:
                    g = out.grad.T
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_sum(self, axis, keepdims, out):
        def backward():
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            g = np.broadcast_to(grad, self.data.shape).copy()
            self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_mean(self, axis, keepdims, out):
        def backward():
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            if axis is None:
                n = self.data.size
            else:
                n = self.data.shape[axis]
            grad = grad / n
            g = np.broadcast_to(grad, self.data.shape).copy()
            self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_exp(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * out.data
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_log(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad / self.data
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_sqrt(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad / (2 * out.data)
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_pow(self, n, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * n * (self.data ** (n - 1))
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_relu(self, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * (self.data > 0).astype(self.data.dtype)
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_sigmoid(self, sig_out, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * sig_out * (1 - sig_out)
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_tanh(self, tanh_out, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * (1 - tanh_out ** 2)
                self.grad = g if self.grad is None else self.grad + g
        return backward
    
    def _make_backward_softmax(self, softmax_out, axis, out):
        def backward():
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            sum_grad = np.sum(out.grad * softmax_out, axis=axis, keepdims=True)
            g = softmax_out * (out.grad - sum_grad)
            self.grad = g if self.grad is None else self.grad + g
        return backward
    
    # ============================================
    # BACKWARD PRINCIPAL
    # ============================================
    
    def backward(self):
        visited = set()
        def reset_grads(v):
            if v in visited:
                return
            visited.add(v)
            v.grad = None
            for prev in v._prev:
                reset_grads(prev)
        reset_grads(self)
        
        self.grad = np.ones_like(self.data)
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo.append(v)
        build_topo(self)
        
        for v in reversed(topo):
            if v.requires_grad:
                v._backward()
    
    def zero_grad(self):
        self.grad = None
    
    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def astype(self, dtype):
        """Convierte el tensor a otro dtype"""
        return Tensor(self.data.astype(dtype), 
                     requires_grad=self.requires_grad)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, grad={self.grad is not None})"
    
    def __len__(self):
        return len(self.data)


# ============================================
# FUNCIONES DE CREACIÓN
# ============================================

def tensor(data, requires_grad=False, dtype=None):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)

def zeros(shape, requires_grad=False):
    return Tensor(np.zeros(shape, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)

def ones(shape, requires_grad=False):
    return Tensor(np.ones(shape, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)

def randn(*shape, requires_grad=False):
    return Tensor(np.random.randn(*shape).astype(DEFAULT_DTYPE), requires_grad=requires_grad)

def rand(*shape, requires_grad=False):
    return Tensor(np.random.rand(*shape).astype(DEFAULT_DTYPE), requires_grad=requires_grad)

def arange(start, stop, step=1, requires_grad=False):
    return Tensor(np.arange(start, stop, step, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)

def eye(n, requires_grad=False):
    return Tensor(np.eye(n, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)
