"""
Sandbox REAL con subprocess aislado + límites + audit log
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from security.permissions import Permission, PermissionSet, is_dangerous
from security.limits import ResourceLimits, LIMITS_STANDARD
from security.process import IsolatedProcess, ProcessResult


@dataclass
class SandboxResult:
    """Resultado de ejecución en sandbox"""
    success: bool
    result: Any = None
    error: str = None
    blocked: bool = False
    reason: str = None
    duration_ms: float = 0.0


class Sandbox:
    """
    Sandbox que controla la ejecución de herramientas.
    
    Ahora soporta:
    - Permission gate (compatible con el código anterior)
    - Subprocess aislado (nuevo)
    - Límites de recursos
    - Audit log persistente
    """
    
    def __init__(self, permissions=None, strict: bool = True,
                 limits: ResourceLimits = None, audit_file: str = None):
        # Normalizar permisos
        if permissions is None:
            self.permissions = PermissionSet()
        elif isinstance(permissions, PermissionSet):
            self.permissions = permissions
        elif isinstance(permissions, set):
            self.permissions = PermissionSet(permissions.copy())
        else:
            raise TypeError(f"permissions debe ser set o PermissionSet, es {type(permissions)}")
        
        self.strict = strict
        self.limits = limits or LIMITS_STANDARD
        self.audit_log = []
        self.approved_dangerous = set()
        self.audit_file = audit_file or "logs/sandbox_audit.json"
        
        # Crear directorio de logs
        os.makedirs(os.path.dirname(self.audit_file) or ".", exist_ok=True)
    
    # ============================================
    # PERMISSIONS (compatible)
    # ============================================
    
    def request_permission(self, permission: Permission) -> bool:
        if self.permissions.has(permission):
            return True
        
        if self.strict and is_dangerous(permission):
            if permission not in self.approved_dangerous:
                self._log("DENIED", f"Permiso peligroso no aprobado: {permission.value}")
                return False
        
        self.permissions.add(permission)
        self._log("GRANTED", f"Permiso otorgado: {permission.value}")
        return True
    
    def approve_dangerous(self, permission: Permission):
        self.approved_dangerous.add(permission)
        self.permissions.add(permission)
        self._log("APPROVED", f"Permiso peligroso aprobado: {permission.value}")
    
    # ============================================
    # EXECUTE - FUNCIÓN PYTHON (compatible)
    # ============================================
    
    def execute(self, func: Callable, permission: Permission, *args, **kwargs) -> SandboxResult:
        """Ejecuta una función Python (con permission gate)"""
        start = time.time()
        
        # Verificar permiso
        if not self.permissions.has(permission):
            self._log("BLOCKED", f"Sin permiso {permission.value}")
            return SandboxResult(
                success=False,
                blocked=True,
                reason=f"Permiso requerido: {permission.value}"
            )
        
        # Verificar peligrosidad en modo estricto
        if self.strict and is_dangerous(permission):
            if permission not in self.approved_dangerous:
                self._log("BLOCKED", f"Permiso peligroso sin aprobar: {permission.value}")
                return SandboxResult(
                    success=False,
                    blocked=True,
                    reason=f"Permiso peligroso requiere aprobación: {permission.value}"
                )
        
        # Ejecutar en el proceso actual (para tools simples)
        try:
            self._log("EXECUTE", f"{func.__name__} con {permission.value}")
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            self._log("SUCCESS", f"{func.__name__}")
            return SandboxResult(success=True, result=result, duration_ms=duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            self._log("ERROR", f"{func.__name__}: {e}")
            return SandboxResult(success=False, error=str(e), duration_ms=duration)
    
    # ============================================
    # EXECUTE_CODE - SUBPROCESS AISLADO (nuevo)
    # ============================================
    
    def execute_code(self, code: str, permission: Permission = None) -> SandboxResult:
        """
        Ejecuta código Python en un subprocess aislado.
        
        Args:
            code: código Python
            permission: permiso requerido (default: EXECUTE_SUBPROCESS)
        
        Returns:
            SandboxResult con stdout, stderr, exit code, etc.
        """
        from security.permissions import Permission as P
        
        if permission is None:
            permission = P.EXECUTE_SUBPROCESS
        
        start = time.time()
        
        # Verificar permiso
        if not self.permissions.has(permission):
            self._log("BLOCKED", f"Sin permiso {permission.value}")
            return SandboxResult(
                success=False,
                blocked=True,
                reason=f"Permiso requerido: {permission.value}"
            )
        
        # Ejecutar en subprocess aislado
        self._log("EXECUTE_CODE", f"subprocess: {code[:50]}...")
        
        result = IsolatedProcess.run_code(code, limits=self.limits)
        
        duration = (time.time() - start) * 1000
        
        if result.success:
            self._log("SUCCESS_CODE", f"exit_code={result.exit_code}, duration={result.duration_ms:.1f}ms")
        else:
            self._log("ERROR_CODE", result.error or result.stderr[:100])
        
        return SandboxResult(
            success=result.success,
            result=result.output,
            error=result.error,
            duration_ms=duration,
        )
    
    # ============================================
    # AUDIT LOG
    # ============================================
    
    def _log(self, action: str, message: str):
        """Registra en el audit log"""
        entry = {
            'timestamp': time.time(),
            'action': action,
            'message': message,
        }
        self.audit_log.append(entry)
    
    def get_audit_log(self, limit: int = 20):
        return self.audit_log[-limit:]
    
    def save_audit(self):
        """Guarda el audit log en disco"""
        try:
            with open(self.audit_file, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception:
            pass
    
    def load_audit(self):
        """Carga el audit log desde disco"""
        try:
            with open(self.audit_file, 'r') as f:
                self.audit_log = json.load(f)
        except Exception:
            self.audit_log = []
    
    def __repr__(self):
        return f"Sandbox(perms={len(self.permissions.permissions)}, strict={self.strict})"


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO SANDBOX V3.0 (REAL)")
    print("="*60)
    
    from security.permissions import Permission, PERMISSIONS_STANDARD
    
    sandbox = Sandbox(
        permissions=PERMISSIONS_STANDARD.copy(),
        strict=False,
        audit_file="logs/test_sandbox_audit.json",
    )
    
    # Test 1: Función Python simple
    print("\n1. Función Python (permission gate):")
    def suma(a, b):
        return a + b
    
    result = sandbox.execute(suma, Permission.EXECUTE_MATH, 5, 3)
    print(f"   Success: {result.success}")
    print(f"   Result: {result.result}")
    
    # Test 2: Código en subprocess
    print("\n2. Código en subprocess aislado:")
    sandbox.approve_dangerous(Permission.EXECUTE_SUBPROCESS)
    result = sandbox.execute_code("print(2 + 2)")
    print(f"   Success: {result.success}")
    print(f"   Result: {result.result}")
    
    # Test 3: Código peligroso (timeout)
    print("\n3. Código con loop infinito (debe timeout):")
    result = sandbox.execute_code("while True: pass")
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error}")
    
    # Test 4: Audit log
    print("\n4. Audit log:")
    for entry in sandbox.get_audit_log(5):
        print(f"   [{entry['action']}] {entry['message'][:60]}")
    
    sandbox.save_audit()
    print(f"\n💾 Audit guardado en {sandbox.audit_file}")
    
    print("\n✅ SANDBOX V3.0 FUNCIONANDO")
