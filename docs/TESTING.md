# 🧪 Guía de Testing

## Ejecutar todos los tests

```bash
pytest tests/ -v
```

Ejecutar un test específico

```bash
pytest tests/test_tensor.py -v
pytest tests/test_security.py::test_sandbox_allows_safe -v
```

Categorías de tests

1. Tests de Tensor (8 tests)

tests/test_tensor.py

· Creación
· Operaciones básicas
· Matmul
· Reducciones
· Activaciones
· Funciones matemáticas
· Dominios (log, sqrt)

2. Gradient Checking (25 tests)

tests/test_gradient_check.py
Verifica el autograd contra diferencias finitas.
Esta es la prueba más importante del proyecto.

3. Tests de Capas (7 tests)

tests/test_layers.py

· Linear
· Sequential
· Flatten
· Aprendizaje XOR

4. Tests de Pérdidas (6 tests)

tests/test_losses.py

· MSE
· BCE
· CrossEntropy

5. Tests de Optimizadores (4 tests)

tests/test_optimizers.py

· SGD
· Adam
· Momentum

6. Tests de Modelos (3 tests)

tests/test_models.py

· XOR end-to-end
· Save/Load
· Gradient flow

7. Tests de Broadcasting (10 tests)

tests/test_broadcasting.py

· Escalar
· Fila/Columna
· 2D/3D
· Capa lineal

8. Tests de Numérica (14 tests)

tests/test_numerics.py

· Softmax estable
· Sigmoid estable
· Dominios
· Float32

9. Tests de Seguridad (15 tests)

tests/test_security.py

· Permisos
· Sandbox
· Parser
· Tool Registry

Gradient checking

El gradient checking verifica que el autograd es matemáticamente correcto:

```python
def numerical_gradient(func, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy(); x_plus.flat[i] += eps
        x_minus = x.copy(); x_minus.flat[i] -= eps
        grad.flat[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad
```

Compara el gradiente analítico con el numérico.
Si difieren más de 1e-4, hay un bug.

CI/CD

.github/workflows/tests.yml ejecuta todos los tests en cada push.

Cobertura objetivo

Módulo Cobertura
core/tensor.py 100%
core/layers.py 100%
core/losses.py 100%
core/optimizers.py 100%
security/* 100%
tools/registry.py 100%

Añadir un nuevo test

1. Crea tests/test_mi_modulo.py
2. Sigue el patrón de otros tests
3. Ejecuta pytest tests/test_mi_modulo.py -v
4. Asegúrate de que todos los tests pasan

Reglas

· No añadir una funcionalidad sin test
· Siempre verificar gradientes si es autograd
· Siempre verificar dominios si es numérico
· Documentar cada test con un docstring claro
  EOF
