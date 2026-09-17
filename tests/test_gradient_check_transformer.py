"""
Gradient checking END-TO-END del Transformer completo
Verifica que los gradientes analíticos coinciden con diferencias finitas.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor
from models.transformer import Transformer


def numerical_gradient_scalar(func, x, eps=1e-5):
    """
    Gradiente numérico de una función escalar respecto a un array.
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


def check_param_gradient(name, model, input_data, param_idx, tol=1e-3):
    """
    Verifica el gradiente de UN parámetro específico del modelo.
    
    1. Corre forward + backward → gradiente analítico
    2. Corre diferencias finitas → gradiente numérico
    3. Compara
    """
    # Solo verificamos unos pocos elementos del parámetro (los demás son costosos)
    params = model.parameters()
    param = params[param_idx]
    shape = param.data.shape
    
    # Resetear grads
    for p in params:
        p.grad = None
    
    # Forward + backward
    x = Tensor(input_data.copy())
    logits = model.forward(x)
    loss = logits.sum()
    loss.backward()
    
    analytic_grad = param.grad
    if analytic_grad is None:
        print(f"   ❌ {name}: gradiente analítico es None")
        return False
    
    # Verificar unos cuantos elementos (no todos, sería muy lento)
    # Tomamos 5 elementos aleatorios
    np.random.seed(42)
    total_size = param.data.size
    sample_size = min(5, total_size)
    flat_indices = np.random.choice(total_size, size=sample_size, replace=False)
    
    param_data = param.data.copy()
    analytic_flat = analytic_grad.flatten()
    
    max_diff = 0.0
    for flat_idx in flat_indices:
        # Convertir índice plano a multi-índice
        multi_idx = np.unravel_index(flat_idx, shape)
        
        # Función para diferencias finitas
        def loss_fn(x_val):
            # Modificar temporalmente el parámetro
            old_val = param_data[multi_idx]
            param_data[multi_idx] = x_val
            param.data = param_data.copy()
            
            # Forward
            logits = model.forward(Tensor(input_data.copy()))
            loss_val = np.sum(logits.data)
            
            # Restaurar
            param_data[multi_idx] = old_val
            param.data = param_data.copy()
            
            return loss_val
        
        # Gradiente numérico
        old_val = param_data[multi_idx]
        eps = 1e-4
        
        param_data[multi_idx] = old_val + eps
        param.data = param_data.copy()
        f_plus = np.sum(model.forward(Tensor(input_data.copy())).data)
        
        param_data[multi_idx] = old_val - eps
        param.data = param_data.copy()
        f_minus = np.sum(model.forward(Tensor(input_data.copy())).data)
        
        param_data[multi_idx] = old_val
        param.data = param_data.copy()
        
        numeric_grad = (f_plus - f_minus) / (2 * eps)
        analytic_val = analytic_flat[flat_idx]
        
        diff = abs(numeric_grad - analytic_val)
        max_diff = max(max_diff, diff)
    
    # Tolerancia relativa (los valores pueden ser grandes)
    analytic_max = np.max(np.abs(analytic_flat))
    if analytic_max > 0:
        rel_diff = max_diff / (analytic_max + 1e-8)
    else:
        rel_diff = max_diff
    
    if max_diff < tol or rel_diff < tol:
        print(f"   ✅ {name}: gradiente correcto (diff={max_diff:.6f})")
        return True
    else:
        print(f"   ❌ {name}: gradiente NO coincide (max_diff={max_diff:.6f}, rel={rel_diff:.6f})")
        return False


def run_gradient_check():
    print("🧪 GRADIENT CHECKING END-TO-END - TRANSFORMER")
    print("="*70)
    print("NOTA: Verificamos solo 5 elementos por parámetro (costoso)")
    print("="*70)
    
    np.random.seed(42)
    
    # Crear modelo pequeño (para que el gradient check sea rápido)
    model = Transformer(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_len=4,
    )
    
    # Input pequeño
    input_data = np.array([[1, 2, 3]])
    
    print(f"\n🔧 Configuración:")
    print(f"   Vocab: {model.vocab_size}")
    print(f"   d_model: {model.d_model}")
    print(f"   num_heads: {model.num_heads}")
    print(f"   num_layers: {model.num_layers}")
    print(f"   Parámetros: {len(model.parameters())}")
    print()
    
    # Verificar cada parámetro
    passed = 0
    total = 0
    
    param_names = []
    for i in range(len(model.parameters())):
        param_names.append(f"param_{i}")
    
    # Nombres más descriptivos
    names = ['token_emb', 'pos_emb', 'lm_head', 'ln_final_gamma', 'ln_final_beta']
    # Bloques
    for b in range(model.num_layers):
        names.extend([
            f'block{b}_attn_Wq', f'block{b}_attn_Wk', f'block{b}_attn_Wv', f'block{b}_attn_Wo',
            f'block{b}_ff_W1', f'block{b}_ff_b1', f'block{b}_ff_W2', f'block{b}_ff_b2',
            f'block{b}_ln1_gamma', f'block{b}_ln1_beta',
            f'block{b}_ln2_gamma', f'block{b}_ln2_beta',
        ])
    
    for i, name in enumerate(names):
        total += 1
        try:
            if check_param_gradient(name, model, input_data, i):
                passed += 1
        except Exception as e:
            print(f"   ❌ {name}: EXCEPTION {e}")
    
    print()
    print("="*70)
    print(f"📊 RESULTADOS: {passed}/{total} parámetros con gradiente correcto")
    print("="*70)
    
    if passed == total:
        print("🎉 ¡TODOS LOS GRADIENTES DEL TRANSFORMER SON CORRECTOS!")
        print("   El autograd es REAL de extremo a extremo.")
        return True
    else:
        print(f"❌ {total - passed} parámetros con gradiente incorrecto")
        return False


if __name__ == "__main__":
    success = run_gradient_check()
    exit(0 if success else 1)
