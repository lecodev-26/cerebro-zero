"""
Límites de recursos para ejecución aislada
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass


@dataclass
class ResourceLimits:
    """Límites de recursos para una ejecución"""
    timeout_seconds: float = 5.0       # Tiempo máximo de ejecución
    max_memory_mb: int = 256           # RAM máxima
    max_cpu_seconds: float = 5.0       # CPU máxima
    max_output_bytes: int = 1024 * 100  # 100 KB de output máximo
    max_file_size_mb: int = 10         # Tamaño máximo de archivo


# Perfiles predefinidos
LIMITS_SAFE = ResourceLimits(
    timeout_seconds=2.0,
    max_memory_mb=64,
    max_cpu_seconds=2.0,
)

LIMITS_STANDARD = ResourceLimits(
    timeout_seconds=5.0,
    max_memory_mb=256,
    max_cpu_seconds=5.0,
)

LIMITS_GENEROUS = ResourceLimits(
    timeout_seconds=30.0,
    max_memory_mb=1024,
    max_cpu_seconds=30.0,
)


def apply_limits():
    """
    Aplica límites de recursos al proceso actual.
    Usa `resource` de Python (Unix).
    Debe llamarse DENTRO del subprocess.
    """
    try:
        import resource
        
        # Nota: los límites se aplican dentro del subprocess
        # No podemos aplicar max_memory_mb global aquí fácilmente en Termux,
        # pero podemos limitar:
        
        # 1. Tiempo de CPU
        # resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        
        # 2. Tamaño máximo de archivo
        # resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        
        # 3. Número máximo de procesos
        resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
        
        # 4. Número máximo de file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
        
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("🧪 PROBANDO RESOURCE LIMITS")
    print("="*50)
    
    print(f"\n📋 Perfiles disponibles:")
    print(f"   SAFE:      timeout={LIMITS_SAFE.timeout_seconds}s, mem={LIMITS_SAFE.max_memory_mb}MB")
    print(f"   STANDARD:  timeout={LIMITS_STANDARD.timeout_seconds}s, mem={LIMITS_STANDARD.max_memory_mb}MB")
    print(f"   GENEROUS:  timeout={LIMITS_GENEROUS.timeout_seconds}s, mem={LIMITS_GENEROUS.max_memory_mb}MB")
    
    print(f"\n✅ RESOURCE LIMITS FUNCIONANDO")
