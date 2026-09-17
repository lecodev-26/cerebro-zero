"""
Gradient checking para las nuevas operaciones del Tensor V3.0
Verifica: gather, embedding, batch_matmul, stack, concat, where, mask
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor


def numerical_gradient(func, x, eps=1e-6):
    """Gradiente numérico usando diferencias finitas centrales"""
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
    """Verifica gradientes de una operación"""
    tensors = [Tensor(inp.copy(), requires_grad=True) for inp in inputs]
    output = op_func(*tensors)
    output.backward()
    
    analytic_grads = []
    for t in tensors:
        if t.grad is None:
            analytic_grads.append(np.zeros_like(t.data))
        else:
            analytic_grads.append(t.grad.copy())
    
    numeric_grads = []
    for i, t in enumerate(tensors):
        def loss_fn(x):
            new_inputs = [inp.copy() for inp in inputs]
            new_inputs[i] = x
            new_tensors = [Tensor(inp, requires_grad=False) for inp in new_inputs]
            out = op_func(*new_tensors)
            return np.sum(out.data)
        
        ng = numerical_gradient(loss_fn, t.data.copy(), eps=eps)
        numeric_grads.append(ng)
    
    for i, (analytic, numeric) in enumerate(zip(analytic_grads, numeric_grads)):
        if analytic.shape != numeric.shape:
            print(f"   ❌ {name} (input {i}): shape mismatch {analytic.shape} vs {numeric.shape}")
            return False
        
        diff = np.max(np.abs(analytic - numeric))
        if diff > tol:
            print(f"   ❌ {name} (input {i}): gradientes NO coinciden (diff={diff})")
            print(f"      Analítico: {analytic.flatten()[:5]}")
            print(f"      Numérico:  {numeric.flatten()[:5]}")
            return False
    
    print(f"   ✅ {name}")
    return True


def run_all():
    print("🧪 GRADIENT CHECKING V3.0 - Nuevas operaciones")
    print("="*70)
    
    passed = 0
    total = 0
    
    # 1. GATHER
    total += 1
    try:
        if check_gradient("gather",
                          lambda x: x.gather(np.array([0, 2, 1])),
                          np.random.randn(3, 4)):
            passed += 1
    except Exception as e:
        print(f"   ❌ gather: {e}")
    
    # 2. GATHER con sum
    total += 1
    try:
        if check_gradient("gather + sum",
                          lambda x: x.gather(np.array([0, 2, 1])).sum(),
                          np.random.randn(3, 4)):
            passed += 1
    except Exception as e:
        print(f"   ❌ gather + sum: {e}")
    
    # 3. BATCH MATMUL 3D
    total += 1
    try:
        if check_gradient("batch_matmul (2,3,4)@(2,4,5)",
                          lambda a, b: a.batch_matmul(b).sum(),
                          np.random.randn(2, 3, 4),
                          np.random.randn(2, 4, 5)):
            passed += 1
    except Exception as e:
        print(f"   ❌ batch_matmul: {e}")
    
    # 4. BATCH MATMUL con broadcasting
    total += 1
    try:
        if check_gradient("batch_matmul (1,3,4)@(2,4,5)",
                          lambda a, b: a.batch_matmul(b).sum(),
                          np.random.randn(1, 3, 4),
                          np.random.randn(2, 4, 5)):
            passed += 1
    except Exception as e:
        print(f"   ❌ batch_matmul broadcast: {e}")
    
    # 5. STACK
    total += 1
    try:
        if check_gradient("stack",
                          lambda a, b: Tensor.stack([a, b], axis=0).sum(),
                          np.random.randn(3, 4),
                          np.random.randn(3, 4)):
            passed += 1
    except Exception as e:
        print(f"   ❌ stack: {e}")
    
    # 6. CONCAT
    total += 1
    try:
        if check_gradient("concat",
                          lambda a, b: Tensor.concat([a, b], axis=0).sum(),
                          np.random.randn(3, 4),
                          np.random.randn(2, 4)):
            passed += 1
    except Exception as e:
        print(f"   ❌ concat: {e}")
    
    # 7. WHERE
    total += 1
    try:
        cond = np.array([[True, False], [False, True]])
        if check_gradient("where",
                          lambda a, b: a.where(cond, b).sum(),
                          np.random.randn(2, 2),
                          np.random.randn(2, 2)):
            passed += 1
    except Exception as e:
        print(f"   ❌ where: {e}")
    
    # 8. MASK
    total += 1
    try:
        mask = np.array([[1.0, 0.0], [0.0, 1.0]])
        if check_gradient("mask",
                          lambda a: a.mask(mask).sum(),
                          np.random.randn(2, 2)):
            passed += 1
    except Exception as e:
        print(f"   ❌ mask: {e}")
    
    # 9. MAX
    total += 1
    try:
        if check_gradient("max (axis=1)",
                          lambda a: a.max(axis=1).sum(),
                          np.random.randn(3, 4)):
            passed += 1
    except Exception as e:
        print(f"   ❌ max: {e}")
    
    # 10. EMBEDDING (via gather + weight)
    # NOTA: indices NO son diferenciables, solo weight
    total += 1
    try:
        indices = np.array([0, 2, 1, 0])
        weight_data = np.random.randn(3, 4)
        
        # Solo verificamos el gradiente del weight
        weight = Tensor(weight_data.copy(), requires_grad=True)
        out = weight.gather(indices)
        loss = out.sum()
        loss.backward()
        
        analytic = weight.grad.copy()
        
        # Gradiente numérico
        def loss_fn(w):
            new_weight = Tensor(w, requires_grad=False)
            out = new_weight.gather(indices)
            return np.sum(out.data)
        
        numeric = numerical_gradient(loss_fn, weight_data.copy())
        
        if np.max(np.abs(analytic - numeric)) < 1e-4:
            print(f"   ✅ embedding (gather de weight)")
            passed += 1
        else:
            print(f"   ❌ embedding: gradientes no coinciden")
    except Exception as e:
        print(f"   ❌ embedding: {e}")
    
    print("="*70)
    print(f"📊 RESULTADOS: {passed}/{total} passed")
    print("="*70)
    
    if passed == total:
        print("🎉 ¡TODAS LAS NUEVAS OPERACIONES PASAN GRADIENT CHECKING!")
        return True
    else:
        print(f"❌ {total - passed} operaciones fallaron")
        return False


if __name__ == "__main__":
    success = run_all()
    exit(0 if success else 1)
