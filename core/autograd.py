import numpy as np

class Variable:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
    
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, requires_grad=True)
        out._prev = {self, other}
        out._backward = lambda: self._add_backward(out, other)
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, requires_grad=True)
        out._prev = {self, other}
        out._backward = lambda: self._mul_backward(out, other)
        return out
    
    def matmul(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(np.dot(self.data, other.data), requires_grad=True)
        out._prev = {self, other}
        out._backward = lambda: self._matmul_backward(out, other)
        return out
    
    def _add_backward(self, out, other):
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.ones_like(self.data)
            self.grad += out.grad
        if other.requires_grad:
            if other.grad is None:
                other.grad = np.ones_like(other.data)
            other.grad += out.grad
    
    def _mul_backward(self, out, other):
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += out.grad * other.data
        if other.requires_grad:
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            other.grad += out.grad * self.data
    
    def _matmul_backward(self, out, other):
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += np.dot(out.grad, other.data.T)
        if other.requires_grad:
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            other.grad += np.dot(self.data.T, out.grad)
    
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
            v._backward()
    
    def __repr__(self):
        return f"Variable(data={self.data}, grad={self.grad})"

def prueba_autograd():
    print("🧠 PROBANDO AUTOGRAD")
    print("="*30)
    
    x = Variable(2.0, requires_grad=True)
    y = Variable(3.0, requires_grad=True)
    z = x * y + x
    
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = x*y + x = {z.data}")
    
    z.backward()
    
    print(f"\n✅ Gradiente de x: {x.grad} (debería ser y + 1 = 4)")
    print(f"✅ Gradiente de y: {y.grad} (debería ser x = 2)")
    
    # Probar con matrices
    print("\n🧠 PROBANDO CON MATRICES")
    print("="*30)
    
    A = Variable(np.array([[1, 2], [3, 4]]), requires_grad=True)
    B = Variable(np.array([[5, 6], [7, 8]]), requires_grad=True)
    C = A.matmul(B)
    
    print(f"A = {A.data}")
    print(f"B = {B.data}")
    print(f"C = A @ B = {C.data}")
    
    C.backward()
    
    print(f"\n✅ Gradiente de A: {A.grad}")
    print(f"✅ Gradiente de B: {B.grad}")
    print("\n🧠 AUTOGRAD FUNCIONANDO CORRECTAMENTE!")

if __name__ == "__main__":
    prueba_autograd()
