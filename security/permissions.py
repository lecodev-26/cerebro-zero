"""
Sistema de permisos para herramientas
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Set


class Permission(Enum):
    """Permisos atómicos que puede tener una herramienta"""
    # Permisos de lectura
    READ_MEMORY = "read_memory"
    READ_FILESYSTEM = "read_filesystem"
    READ_NETWORK = "read_network"
    READ_SYSTEM_INFO = "read_system_info"
    
    # Permisos de escritura
    WRITE_MEMORY = "write_memory"
    WRITE_FILESYSTEM = "write_filesystem"
    WRITE_NETWORK = "write_network"
    
    # Permisos de ejecución
    EXECUTE_MATH = "execute_math"
    EXECUTE_SUBPROCESS = "execute_subprocess"
    EXECUTE_PYTHON = "execute_python"
    
    # Permisos especiales (peligrosos)
    DELETE_FILES = "delete_files"
    MODIFY_SYSTEM = "modify_system"
    SEND_EMAIL = "send_email"
    MAKE_PAYMENT = "make_payment"


# Grupos de permisos predefinidos
PERMISSIONS_SAFE = {
    Permission.READ_MEMORY,
    Permission.WRITE_MEMORY,
    Permission.EXECUTE_MATH,
}

PERMISSIONS_STANDARD = PERMISSIONS_SAFE | {
    Permission.READ_SYSTEM_INFO,
    Permission.READ_FILESYSTEM,
}

PERMISSIONS_NETWORK = PERMISSIONS_STANDARD | {
    Permission.READ_NETWORK,
}

PERMISSIONS_FULL = set(Permission)


@dataclass
class PermissionSet:
    """Conjunto de permisos con verificación"""
    permissions: Set[Permission] = field(default_factory=set)
    
    def has(self, permission: Permission) -> bool:
        return permission in self.permissions
    
    def has_all(self, *permissions: Permission) -> bool:
        return all(p in self.permissions for p in permissions)
    
    def has_any(self, *permissions: Permission) -> bool:
        return any(p in self.permissions for p in permissions)
    
    def add(self, permission: Permission):
        self.permissions.add(permission)
    
    def remove(self, permission: Permission):
        self.permissions.discard(permission)
    
    def __contains__(self, permission):
        return self.has(permission)
    
    def __repr__(self):
        return f"PermissionSet({len(self.permissions)} permisos)"


# Permisos por defecto según peligrosidad
DANGEROUS_PERMISSIONS = {
    Permission.EXECUTE_SUBPROCESS,
    Permission.EXECUTE_PYTHON,
    Permission.DELETE_FILES,
    Permission.MODIFY_SYSTEM,
    Permission.SEND_EMAIL,
    Permission.MAKE_PAYMENT,
}


def is_dangerous(permission: Permission) -> bool:
    """Verifica si un permiso es peligroso"""
    return permission in DANGEROUS_PERMISSIONS


if __name__ == "__main__":
    print("🧪 PROBANDO SISTEMA DE PERMISOS")
    print("="*50)
    
    # Crear conjunto de permisos seguros
    ps = PermissionSet(PERMISSIONS_SAFE.copy())
    print(f"Permisos seguros: {len(ps.permissions)}")
    print(f"  ¿Tiene EXECUTE_MATH? {ps.has(Permission.EXECUTE_MATH)}")
    print(f"  ¿Tiene READ_NETWORK? {ps.has(Permission.READ_NETWORK)}")
    print(f"  ¿Tiene EXECUTE_SUBPROCESS? {ps.has(Permission.EXECUTE_SUBPROCESS)}")
    
    # Verificar peligrosidad
    print(f"\n  ¿EXECUTE_SUBPROCESS es peligroso? {is_dangerous(Permission.EXECUTE_SUBPROCESS)}")
    print(f"  ¿EXECUTE_MATH es peligroso? {is_dangerous(Permission.EXECUTE_MATH)}")
    
    print("\n✅ SISTEMA DE PERMISOS FUNCIONANDO")
