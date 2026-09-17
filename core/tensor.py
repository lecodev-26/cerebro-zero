"""
Tensor V3.0 - Autograd completo para Transformer
Añade: gather, embedding, batch matmul, stack, concat, where, mask
"""

import numpy as np

DEFAULT_DTYPE = np.float64


def set_dtype(dtype):
    """Configura el dtype por defecto (float32 o float64)"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype


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
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'neg'
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = -out.grad
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        return out
    
    # ============================================
    # MATMUL (2D)
    # ============================================
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = 'matmul'
        out._backward = self._make_backward_matmul(other, out)
        return out
    
    # ============================================
    # BATCH MATMUL (N-D)
    # ============================================
    
    def batch_matmul(self, other):
        """
        Producto matricial con batch.
        Soporta (..., n, m) @ (..., m, p) -> (..., n, p)
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.matmul(self.data, other.data)
        out = Tensor(out_data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = 'batch_matmul'
        out._backward = self._make_backward_batch_matmul(other, out)
        return out
    
    # ============================================
    # GATHER (indexado diferenciable)
    # ============================================
    
    def gather(self, indices):
        """
        Indexa el tensor por la primera dimensión.
        indices: array de índices
        out[i] = self[indices[i]]
        """
        indices = np.array(indices, dtype=int)
        out_data = self.data[indices]
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'gather'
        out._backward = self._make_backward_gather(indices, out)
        return out
    
    # ============================================
    # EMBEDDING
    # ============================================
    
    def embedding(self, vocab_size: int, embed_dim: int):
        """
        Crea una capa de embedding y aplica self como índices.
        self debe ser un tensor de índices enteros.
        """
        # Inicializar pesos del embedding
        lim = 1.0 / np.sqrt(embed_dim)
        weight_data = np.random.uniform(-lim, lim, (vocab_size, embed_dim))
        weight = Tensor(weight_data, requires_grad=True)
        
        # Aplicar gather
        indices = self.data.astype(int)
        out = weight.gather(indices)
        out._op = 'embedding'
        
        # Guardar referencia al peso para el grafo
        out._embedding_weight = weight
        
        return out, weight
    
    # ============================================
    # STACK / CONCAT
    # ============================================
    
    @staticmethod
    def stack(tensors, axis=0):
        """Apila tensores a lo largo de un nuevo eje"""
        if not tensors:
            raise ValueError("Lista vacía")
        
        data = np.stack([t.data for t in tensors], axis=axis)
        out = Tensor(data, requires_grad=any(t.requires_grad for t in tensors))
        out._op = 'stack'
        out._backward = Tensor._make_backward_stack(tensors, axis, out)
        return out
    
    @staticmethod
    def concat(tensors, axis=0):
        """Concatena tensores a lo largo de un eje existente"""
        if not tensors:
            raise ValueError("Lista vacía")
        
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(data, requires_grad=any(t.requires_grad for t in tensors))
        out._op = 'concat'
        out._backward = Tensor._make_backward_concat(tensors, axis, out)
        return out
    
    # ============================================
    # WHERE / MASK
    # ============================================
    
    def where(self, condition, other):
        """
        where(condition, self, other)
        Devuelve self donde condition=True, other donde condition=False
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        cond = np.array(condition)
        out_data = np.where(cond, self.data, other.data)
        out = Tensor(out_data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other))
        out._op = 'where'
        out._backward = self._make_backward_where(cond, other, out)
        return out
    
    def mask(self, mask_array):
        """
        Aplica una máscara: donde mask=0, el tensor se pone a 0.
        """
        mask_array = np.array(mask_array)
        out_data = self.data * mask_array
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'mask'
        out._backward = self._make_backward_mask(mask_array, out)
        return out
    
    # ============================================
    # RESHAPE / TRANSPOSE
    # ============================================
    
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
    
    def permute(self, *dims):
        """Alias de transpose"""
        return self.transpose(*dims)
    
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
        if self.data.size == 0:
            raise ValueError("mean() no funciona en tensor vacío")
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'mean'
        out._backward = self._make_backward_mean(axis, keepdims, out)
        return out
    
    def max(self, axis=None, keepdims=False):
        """Max con backward (gradiente solo al máximo)"""
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,))
        out._op = 'max'
        out._backward = self._make_backward_max(axis, keepdims, out)
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
        data = self.data
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
    # HELPER: REDUCIR GRADIENTE
    # ============================================
    
    def _reduce_grad(self, grad, original_shape):
        if grad.shape == original_shape:
            return grad
        
        while len(grad.shape) > len(original_shape):
            grad = np.sum(grad, axis=0)
        
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
    
    def _make_backward_batch_matmul(self, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                # (..., n, m) @ (..., m, p) -> grad_self = grad_out @ other^T
                g = np.matmul(out.grad, other.data.swapaxes(-2, -1))
                # Reducir broadcasting en batch dims
                g = self._reduce_grad(g, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = np.matmul(self.data.swapaxes(-2, -1), out.grad)
                # Reducir broadcasting en batch dims
                g = other._reduce_grad(g, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_gather(self, indices, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                # Acumular gradientes en los índices correspondientes
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                
                # Reducir out.grad a la forma de indices si es necesario
                grad_to_scatter = out.grad
                # Si out.grad tiene más dimensiones que indices, agregar
                if grad_to_scatter.shape != indices.shape + self.data.shape[1:]:
                    # Sumar ejes extra
                    while len(grad_to_scatter.shape) > len(indices.shape) + len(self.data.shape) - 1:
                        grad_to_scatter = np.sum(grad_to_scatter, axis=0)
                
                # Scatter add
                np.add.at(self.grad, indices, grad_to_scatter)
        return backward
    
    @staticmethod
    def _make_backward_stack(tensors, axis, out):
        def backward():
            if out.grad is None:
                return
            # Dividir out.grad a lo largo del eje
            n = len(tensors)
            splits = np.split(out.grad, n, axis=axis)
            for t, g in zip(tensors, splits):
                if t.requires_grad:
                    # Eliminar el eje añadido por stack
                    g = np.squeeze(g, axis=axis)
                    t.grad = g if t.grad is None else t.grad + g
        return backward
    
    @staticmethod
    def _make_backward_concat(tensors, axis, out):
        def backward():
            if out.grad is None:
                return
            sizes = [t.data.shape[axis] for t in tensors]
            splits = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
            for t, g in zip(tensors, splits):
                if t.requires_grad:
                    t.grad = g if t.grad is None else t.grad + g
        return backward
    
    def _make_backward_where(self, cond, other, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = np.where(cond, out.grad, 0)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = np.where(cond, 0, out.grad)
                other.grad = g if other.grad is None else other.grad + g
        return backward
    
    def _make_backward_mask(self, mask_array, out):
        def backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * mask_array
                self.grad = g if self.grad is None else self.grad + g
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
    
    def _make_backward_max(self, axis, keepdims, out):
        def backward():
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            # El gradiente va solo al máximo
            if axis is None:
                mask = (self.data == out.data)
                g = np.where(mask, out.grad, 0)
            else:
                out_expanded = np.expand_dims(out.data, axis) if not keepdims else out.data
                mask = (self.data == out_expanded)
                out_grad_expanded = np.expand_dims(out.grad, axis) if not keepdims else out.grad
                g = np.where(mask, out_grad_expanded, 0)
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
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)
    
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
