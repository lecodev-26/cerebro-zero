"""
Gradient checking: verifica autograd contra diferencias finitas
Esta es LA prueba más importante del autograd.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor, tensor

def numerical_gradient(func, x, eps=1e-6):
    """
    Calcula el gradiente numérico usando diferencias finitas centrales
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + eps
        f_plus = func(x)
        
        x[idx] = old_val - eps
        f_minus = func(x)
        
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        
        x[idx] = old_val
        it.iternext()
    return grad


def check_gradient(name, op_func, *inputs, eps=1e-6, tol=1e-4):
    """
    Verifica el gradiente de una operación
    """
    # Crear tensores
    tensors = [Tensor(inp.copy(), requires_grad=True) for inp in inputs]
    
    # Forward
    output = op_func(*tensors)
    
    # Backward
    output.backward()
    
    # Gradientes analíticos
    analytic_grads = [t.grad.copy() for t in tensors]
    
    # Gradientes numéricos
    numeric_grads = []
    for i, t in enumerate(tensors):
        def loss_fn(x):
            # Recrear tensores con x modificado en posición i
            new_inputs = [Tensor(inp.copy(), requires_grad=False) for inp in inputs]
            new_inputs[i] = Tensor(x.copy(), requires_grad=False)
            out = op_func(*new_inputs)
            return np.sum(out.data)
        
        ng = numerical_gradient(loss_fn, t.data.copy(), eps=eps)
        numeric_grads.append(ng)
    
    # Comparar
    for i, (analytic, numeric) in enumerate(zip(analytic_grads, numeric_grads)):
        diff = np.abs(analytic - numeric)
        max_diff = np.max(diff)
        
        if max_diff > tol:
            print(f"   ❌ {name} (input {i}): gradientes NO coinciden")
            print(f"      Analítico: {analytic}")
            print(f"      Numérico:  {numeric}")
            print(f"      Max diff:  {max_diff}")
            return False
    
    print(f"   ✅ {name}: gradiente correcto (max diff < {tol})")
    return True


def run_all_tests():
    print("🧪 GRADIENT CHECKING - Verificando autograd")
    print("="*60)
    
    passed = 0
    failed = 0
    
    # 1. Suma
    if check_gradient("suma (a + b)", lambda a, b: a + b, 
                      np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        passed += 1
    else:
        failed += 1
    
    # 2. Resta
    if check_gradient("resta (a - b)", lambda a, b: a - b,
                      np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        passed += 1
    else:
        failed += 1
    
    # 3. Multiplicación
    if check_gradient("multiplicación (a * b)", lambda a, b: a * b,
                      np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        passed += 1
    else:
        failed += 1
    
    # 4. División
    if check_gradient("división (a / b)", lambda a, b: a / b,
                      np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        passed += 1
    else:
        failed += 1
    
    # 5. Matmul
    if check_gradient("matmul (a @ b)", lambda a, b: a.matmul(b),
                      np.array([[1.0, 2.0], [3.0, 4.0]]),
                      np.array([[5.0, 6.0], [7.0, 8.0]])):
        passed += 1
    else:
        failed += 1
    
    # 6. Sum (axis=None)
    if check_gradient("sum (todos)", lambda a: a.sum(),
                      np.array([[1.0, 2.0], [3.0, 4.0]])):
        passed += 1
    else:
        failed += 1
    
    # 7. Sum (axis=0)
    if check_gradient("sum (axis=0)", lambda a: a.sum(axis=0),
                      np.array([[1.0, 2.0], [3.0, 4.0]])):
        passed += 1
    else:
        failed += 1
    
    # 8. Sum (axis=1)
    if check_gradient("sum (axis=1)", lambda a: a.sum(axis=1),
                      np.array([[1.0, 2.0], [3.0, 4.0]])):
        passed += 1
    else:
        failed += 1
    
    # 9. Mean (todos)
    if check_gradient("mean (todos)", lambda a: a.mean(),
                      np.array([[1.0, 2.0], [3.0, 4.0]])):
        passed += 1
    else:
        failed += 1
    
    # 10. Mean (axis=0)
    if check_gradient("mean (axis=0)", lambda a: a.mean(axis=0),
                      np.array([[1.0, 2.0], [3.0, 4.0]])):
        passed += 1
    else:
        failed += 1
    
    # 11. Exp
    if check_gradient("exp", lambda a: a.exp(),
                      np.array([0.5, 1.0, 1.5])):
        passed += 1
    else:
        failed += 1
    
    # 12. Log (corregido)
    if check_gradient("log", lambda a: a.log(),
                      np.array([0.5, 1.0, 1.5])):
        passed += 1
    else:
        failed += 1
    
    # 13. Sqrt (corregido)
    if check_gradient("sqrt", lambda a: a.sqrt(),
                      np.array([0.5, 1.0, 1.5])):
        passed += 1
    else:
        failed += 1
    
    # 14. Pow
    if check_gradient("pow(2)", lambda a: a.pow(2),
                      np.array([1.0, 2.0, 3.0])):
        passed += 1
    else:
        failed += 1
    
    # 15. Pow (3)
    if check_gradient("pow(3)", lambda a: a.pow(3),
                      np.array([1.0, 2.0, 3.0])):
        passed += 1
    else:
        failed += 1
    
    # 16. ReLU
    if check_gradient("relu", lambda a: a.relu(),
                      np.array([-1.0, 0.5, 2.0])):
        passed += 1
    else:
        failed += 1
    
    # 17. Sigmoid
    if check_gradient("sigmoid", lambda a: a.sigmoid(),
                      np.array([-1.0, 0.5, 2.0])):
        passed += 1
    else:
        failed += 1
    
    # 18. Tanh
    if check_gradient("tanh", lambda a: a.tanh(),
                      np.array([-1.0, 0.5, 2.0])):
        passed += 1
    else:
        failed += 1
    
    # 19. Softmax
    if check_gradient("softmax", lambda a: a.softmax(),
                      np.array([1.0, 2.0, 3.0])):
        passed += 1
    else:
        failed += 1
    
    # 20. Reshape
    if check_gradient("reshape", lambda a: a.reshape(3, 2),
                      np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])):
        passed += 1
    else:
        failed += 1
    
    # 21. Transpose
    if check_gradient("transpose", lambda a: a.transpose(),
                      np.array([[1.0, 2.0], [3.0, 4.0]])):
        passed += 1
    else:
        failed += 1
    
    # 22. Broadcasting
    if check_gradient("broadcasting (a + b) con b escalar",
                      lambda a, b: a + b,
                      np.array([[1.0, 2.0], [3.0, 4.0]]),
                      np.array([10.0])):
        passed += 1
    else:
        failed += 1
    
    # 23. Cadena compleja
    def chain(a, b):
        return ((a * b) + a).sum()
    if check_gradient("cadena: ((a*b) + a).sum()", chain,
                      np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        passed += 1
    else:
        failed += 1
    
    # 24. Cadena con matmul y suma
    def chain2(a, b):
        return (a.matmul(b)).sum()
    if check_gradient("cadena: (a @ b).sum()", chain2,
                      np.array([[1.0, 2.0], [3.0, 4.0]]),
                      np.array([[5.0, 6.0], [7.0, 8.0]])):
        passed += 1
    else:
        failed += 1
    
    # 25. Red neuronal mini (Linear → ReLU → Sum)
    def mini_nn(x, w, b):
        return (x.matmul(w) + b).relu().sum()
    if check_gradient("mini NN: relu(x@w + b).sum()", mini_nn,
                      np.array([[1.0, 2.0]]),
                      np.array([[0.5], [0.5]]),
                      np.array([[0.1]])):
        passed += 1
    else:
        failed += 1
    
    print("\n" + "="*60)
    print(f"📊 RESULTADOS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("🎉 ¡TODOS LOS TESTS DE GRADIENTE PASARON!")
        print("   El autograd está verificado correctamente.")
        return True
    else:
        print(f"❌ {failed} tests fallaron. Revisar autograd.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
