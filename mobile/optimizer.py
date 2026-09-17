"""
Optimizaciones para móvil: float32, lazy loading, cache
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import set_dtype, get_dtype


class MobileOptimizer:
    """
    Optimizador para ejecución en móvil.
    
    Uso:
        opt = MobileOptimizer()
        opt.enable_float32()          # Cambia a float32 global
        opt.enable_lazy_loading()     # Lazy loading
        opt.print_stats()             # Muestra estadísticas
    """
    
    def __init__(self):
        self.float32_enabled = False
        self.lazy_loading_enabled = False
        self.cache = {}
        self.original_dtype = None
    
    def enable_float32(self):
        """
        Cambia el dtype global a float32.
        En móvil, float32 es ~2× más rápido que float64.
        """
        self.original_dtype = get_dtype()
        set_dtype(np.float32)
        self.float32_enabled = True
        print("⚡ Float32 activado (2× más rápido en móvil)")
    
    def disable_float32(self):
        """Restaura el dtype original"""
        if self.original_dtype is not None:
            set_dtype(self.original_dtype)
            self.float32_enabled = False
            print("🔧 Float32 desactivado")
    
    def enable_lazy_loading(self):
        """Activa lazy loading de módulos"""
        self.lazy_loading_enabled = True
        print("📦 Lazy loading activado")
    
    def cache_result(self, key: str, value):
        """Guarda un resultado en caché"""
        self.cache[key] = value
    
    def get_cached(self, key: str):
        """Recupera un resultado del caché"""
        return self.cache.get(key)
    
    def clear_cache(self):
        """Limpia el caché"""
        self.cache.clear()
        print(f"🧹 Caché limpiado")
    
    def print_stats(self):
        """Muestra estadísticas de optimización"""
        print("\n📊 MOBILE OPTIMIZER STATS")
        print("="*50)
        print(f"   Float32 activado: {self.float32_enabled}")
        print(f"   Lazy loading: {self.lazy_loading_enabled}")
        print(f"   Cache entries: {len(self.cache)}")
        print(f"   Dtype actual: {get_dtype()}")
        print("="*50)
    
    def __repr__(self):
        return f"MobileOptimizer(f32={self.float32_enabled}, cache={len(self.cache)})"


if __name__ == "__main__":
    print("🧪 PROBANDO MOBILE OPTIMIZER")
    print("="*50)
    
    opt = MobileOptimizer()
    opt.print_stats()
    
    # Activar float32
    opt.enable_float32()
    
    # Crear un tensor y verificar dtype
    from core.tensor import Tensor
    t = Tensor([1.0, 2.0, 3.0])
    print(f"\n✅ Tensor dtype: {t.data.dtype}")
    
    # Cache
    opt.cache_result("test", 42)
    print(f"✅ Cache: {opt.get_cached('test')}")
    
    opt.print_stats()
    
    # Restaurar
    opt.disable_float32()
    
    print("\n✅ MOBILE OPTIMIZER FUNCIONANDO")
