"""
Ejecución de código en subprocess aislado
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import tempfile
import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from security.limits import ResourceLimits, LIMITS_STANDARD


@dataclass
class ProcessResult:
    """Resultado de una ejecución en subprocess"""
    success: bool
    output: Any = None
    stdout: str = ''
    stderr: str = ''
    error: Optional[str] = None
    exit_code: int = -1
    duration_ms: float = 0.0
    timed_out: bool = False


class IsolatedProcess:
    """
    Ejecuta código Python en un subprocess aislado.
    
    Uso:
        result = IsolatedProcess.run_code(
            code="print(2 + 2)",
            limits=LIMITS_STANDARD,
        )
    """
    
    @staticmethod
    def run_code(code: str, limits: ResourceLimits = None,
                 work_dir: str = None) -> ProcessResult:
        """
        Ejecuta código Python en un subprocess aislado.
        
        Args:
            code: código Python a ejecutar
            limits: límites de recursos
            work_dir: directorio de trabajo (default: temp)
        
        Returns:
            ProcessResult
        """
        if limits is None:
            limits = LIMITS_STANDARD
        
        start = time.time()
        
        # Crear directorio temporal aislado
        if work_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="cerebro_sandbox_")
        else:
            tmp_dir = work_dir
        
        # Escribir código en un archivo
        script_path = os.path.join(tmp_dir, "script.py")
        with open(script_path, 'w') as f:
            f.write(code)
        
        try:
            # Ejecutar en subprocess con límites
            # NOTA: En Termux, `subprocess.run` con `timeout` es lo más fiable
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=limits.timeout_seconds,
                cwd=tmp_dir,
                env={
                    'PATH': os.environ.get('PATH', ''),
                    'HOME': tmp_dir,
                    'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                    'PYTHONDONTWRITEBYTECODE': '1',
                },
            )
            
            duration = (time.time() - start) * 1000
            
            stdout = proc.stdout[:limits.max_output_bytes]
            stderr = proc.stderr[:limits.max_output_bytes]
            
            return ProcessResult(
                success=(proc.returncode == 0),
                output=stdout.strip() if proc.returncode == 0 else None,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                duration_ms=duration,
            )
        
        except subprocess.TimeoutExpired:
            duration = (time.time() - start) * 1000
            return ProcessResult(
                success=False,
                error=f"Timeout después de {limits.timeout_seconds}s",
                duration_ms=duration,
                timed_out=True,
            )
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ProcessResult(
                success=False,
                error=str(e),
                duration_ms=duration,
            )


if __name__ == "__main__":
    print("🧪 PROBANDO ISOLATED PROCESS")
    print("="*50)
    
    # Test 1: código simple
    print("\n1. Código simple:")
    result = IsolatedProcess.run_code("print(2 + 2)")
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output}")
    print(f"   Duration: {result.duration_ms:.1f}ms")
    
    # Test 2: código con error
    print("\n2. Código con error:")
    result = IsolatedProcess.run_code("print(1/0)")
    print(f"   Success: {result.success}")
    print(f"   Stderr: {result.stderr[:100]}")
    
    # Test 3: código peligroso (timeout)
    print("\n3. Código con loop infinito (timeout 2s):")
    from security.limits import LIMITS_SAFE
    result = IsolatedProcess.run_code(
        "while True: pass",
        limits=LIMITS_SAFE,
    )
    print(f"   Success: {result.success}")
    print(f"   Timed out: {result.timed_out}")
    print(f"   Duration: {result.duration_ms:.1f}ms")
    
    print("\n✅ ISOLATED PROCESS FUNCIONANDO")
