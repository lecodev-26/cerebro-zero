"""
Sandbox de ejecución para herramientas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Callable, Any
from security.permissions import Permission, PermissionSet, is_dangerous


@dataclass
class SandboxResult:
    """Resultado de ejecución en sandbox"""
    success: bool
    result: Any = None
    error: str = None
    blocked: bool = False
    reason: str = None


class Sandbox:
    """Sandbox que controla la ejecución de herramientas"""
    
    def __init__(self, permissions=None, strict: bool = True):
        # Normalizar: si nos pasan un set, envolverlo en PermissionSet
        if permissions is None:
            self.permissions = PermissionSet()
        elif isinstance(permissions, PermissionSet):
            self.permissions = permissions
        elif isinstance(permissions, set):
            self.permissions = PermissionSet(permissions.copy())
        else:
            raise TypeError(f"permissions debe ser set o PermissionSet, es {type(permissions)}")
        
        self.strict = strict
        self.audit_log = []
        self.approved_dangerous = set()
    
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
    
    def execute(self, func: Callable, permission: Permission, *args, **kwargs) -> SandboxResult:
        if not self.permissions.has(permission):
            self._log("BLOCKED", f"Sin permiso {permission.value}")
            return SandboxResult(
                success=False,
                blocked=True,
                reason=f"Permiso requerido: {permission.value}"
            )
        
        if self.strict and is_dangerous(permission):
            if permission not in self.approved_dangerous:
                self._log("BLOCKED", f"Permiso peligroso sin aprobar: {permission.value}")
                return SandboxResult(
                    success=False,
                    blocked=True,
                    reason=f"Permiso peligroso requiere aprobación: {permission.value}"
                )
        
        try:
            self._log("EXECUTE", f"{func.__name__} con {permission.value}")
            result = func(*args, **kwargs)
            self._log("SUCCESS", f"{func.__name__}")
            return SandboxResult(success=True, result=result)
        except Exception as e:
            self._log("ERROR", f"{func.__name__}: {e}")
            return SandboxResult(success=False, error=str(e))
    
    def _log(self, action: str, message: str):
        self.audit_log.append({'action': action, 'message': message})
    
    def get_audit_log(self, limit: int = 20):
        return self.audit_log[-limit:]
    
    def __repr__(self):
        return f"Sandbox(permisos={len(self.permissions.permissions)}, strict={self.strict})"


if __name__ == "__main__":
    print("🧪 PROBANDO SANDBOX")
    print("="*50)
    
    sandbox = Sandbox(strict=True)
    
    def operacion_segura(a, b):
        return a + b
    
    def operacion_peligrosa(cmd):
        return f"Ejecutando: {cmd}"
    
    print("\n1. Permiso seguro (EXECUTE_MATH):")
    sandbox.permissions.add(Permission.EXECUTE_MATH)
    r = sandbox.execute(operacion_segura, Permission.EXECUTE_MATH, 5, 3)
    print(f"   Success: {r.success}, Result: {r.result}")
    
    print("\n2. Permiso peligroso sin aprobación:")
    r = sandbox.execute(operacion_peligrosa, Permission.EXECUTE_SUBPROCESS, "rm -rf /")
    print(f"   Blocked: {r.blocked}, Reason: {r.reason}")
    
    print("\n3. Permiso peligroso CON aprobación:")
    sandbox.approve_dangerous(Permission.EXECUTE_SUBPROCESS)
    r = sandbox.execute(operacion_peligrosa, Permission.EXECUTE_SUBPROCESS, "ls -la")
    print(f"   Success: {r.success}, Result: {r.result}")
    
    print("\n4. Sandbox con set (no PermissionSet):")
    sandbox2 = Sandbox(permissions={Permission.EXECUTE_MATH})
    r = sandbox2.execute(operacion_segura, Permission.EXECUTE_MATH, 1, 2)
    print(f"   Success: {r.success}, Result: {r.result}")
    
    print("\n✅ SANDBOX FUNCIONANDO")
