# 📚 API Reference

## Tensor

### Creación

```python
from core.tensor import Tensor, tensor, zeros, ones, randn

# Crear desde lista
t = Tensor([1, 2, 3])

# Con gradiente
t = Tensor([1, 2, 3], requires_grad=True)

# Utilidades
z = zeros((2, 3))
o = ones((2, 3))
r = randn(2, 3)
```

Operaciones

```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

a + b         # Suma
a - b         # Resta
a * b         # Multiplicación
a / b         # División
a.matmul(b)   # Producto matricial
a.reshape(3, 1)
a.transpose()
a.sum()
a.sum(axis=0)
a.mean()
a.mean(axis=1)
```

Funciones matemáticas

```python
x = Tensor([1.0, 2.0, 3.0])

x.exp()       # e^x
x.log()       # ln(x) — requiere x > 0
x.sqrt()      # √x — requiere x >= 0
x.pow(2)      # x^2
x.relu()      # max(0, x)
x.sigmoid()   # 1/(1+e^(-x))
x.tanh()      # tanh(x)
x.softmax()   # softmax
```

Backward

```python
x = Tensor([1.0, 2.0], requires_grad=True)
y = (x * 2 + 1).sum()
y.backward()
print(x.grad)  # [2.0, 2.0]
```

Dtype

```python
from core.tensor import set_dtype, get_dtype
import numpy as np

set_dtype(np.float32)  # Modo móvil
set_dtype(np.float64)  # Precisión completa
```

Layers

Linear

```python
from core.layers import Linear

layer = Linear(in_features=3, out_features=2)
x = Tensor([[1, 2, 3]])
y = layer(x)
```

Sequential

```python
from core.layers import Sequential, Linear
from core.activations import ReLU, Sigmoid

model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)

y = model(Tensor([[1, 2]]))
```

Flatten

```python
from core.layers import Flatten

flatten = Flatten()
x = Tensor(np.random.randn(2, 3, 4, 5))
y = flatten(x)  # shape (2, 60)
```

Losses

```python
from core.losses import MSELoss, BCELoss, CrossEntropyLoss

# MSE
loss_fn = MSELoss()
loss = loss_fn(pred, target)

# BCE (clasificación binaria)
loss_fn = BCELoss()
loss = loss_fn(pred, target)

# CrossEntropy (clasificación múltiple)
loss_fn = CrossEntropyLoss()
loss = loss_fn(logits, targets)
```

Optimizers

```python
from core.optimizers import SGD, Adam

# SGD con momentum
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = Adam(model.parameters(), lr=0.001)

# Bucle de entrenamiento
for epoch in range(epochs):
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Security

Permisos

```python
from security.permissions import Permission, PermissionSet, PERMISSIONS_SAFE

ps = PermissionSet(PERMISSIONS_SAFE.copy())
ps.has(Permission.EXECUTE_MATH)  # True
```

Sandbox

```python
from security.sandbox import Sandbox
from security.permissions import Permission

sandbox = Sandbox(strict=True)
sandbox.permissions.add(Permission.EXECUTE_MATH)

def suma(a, b):
    return a + b

result = sandbox.execute(suma, Permission.EXECUTE_MATH, 5, 3)
# SandboxResult(success=True, result=8)
```

Parser

```python
from security.parser import Parser

parser = Parser()
parsed = parser.parse("¿Cuánto es 5 + 3?")
# ParsedCommand(
#   intent='math',
#   entities={'numbers': [5.0, 3.0], 'operator': '+'},
#   dangerous=False
# )
```

Tool Registry

```python
from tools.registry import ToolRegistry
from security.permissions import Permission
from security.sandbox import Sandbox

sandbox = Sandbox()
sandbox.permissions.add(Permission.EXECUTE_MATH)
registry = ToolRegistry(sandbox=sandbox)

@registry.register(
    name="calculadora",
    description="Suma dos números",
    permissions={Permission.EXECUTE_MATH}
)
def calculadora(a, b):
    return a + b

result = registry.execute('calculadora', 5, 3)
# {'success': True, 'result': 8}
```

Memory

```python
from memory.retrieval import MemoryRetrieval

memoria = MemoryRetrieval()
memoria.add_knowledge("nombre", "Manuel")
memoria.add_experience("hola", "Hola, ¿qué tal?", context="saludo")

resultados = memoria.remember("nombre")
```

Continuous Learning

```python
from training.continuous_learning import ContinuousLearning

aprendizaje = ContinuousLearning()
aprendizaje.aprender("hola", "Hola, soy Cerebro Zero", correcto=True)

aprendizaje.estadisticas()
```

