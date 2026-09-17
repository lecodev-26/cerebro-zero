"""
Registro de herramientas con permisos
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Callable, Set, Dict, Optional
from security.permissions import Permission, PermissionSet


@dataclass
class Tool:
    """Definición de una herramienta"""
    name: str
    description: str
    func: Callable
    required_permissions: Set[Permission]
    dangerous: bool = False
    rate_limit: int = 100  # llamadas máximas por minuto
    
    def __repr__(self):
        return f"Tool({self.name}, perms={len(self.required_permissions)})"


class ToolRegistry:
    """Registro de herramientas con verificación de permisos"""
    
    def __init__(self, sandbox=None):
        self.tools: Dict[str, Tool] = {}
        self.sandbox = sandbox
    
    def register(self, name: str, description: str, 
                 permissions: Set[Permission], dangerous: bool = False):
        """Decorador para registrar herramientas"""
        def decorator(func):
            tool = Tool(
                name=name,
                description=description,
                func=func,
                required_permissions=permissions,
                dangerous=dangerous,
            )
            self.tools[name] = tool
            print(f"📦 Herramienta registrada: {name}")
            return func
        return decorator
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    def list_tools(self) -> list:
        return [
            {
                'name': t.name,
                'description': t.description,
                'permissions': [p.value for p in t.required_permissions],
                'dangerous': t.dangerous,
            }
            for t in self.tools.values()
        ]
    
    def execute(self, name: str, *args, **kwargs):
        """Ejecuta una herramienta verificando permisos"""
        tool = self.tools.get(name)
        if not tool:
            return {'success': False, 'error': f'Herramienta no existe: {name}'}
        
        # Verificar todos los permisos
        if self.sandbox:
            for perm in tool.required_permissions:
                if not self.sandbox.permissions.has(perm):
                    return {
                        'success': False,
                        'error': f'Permiso faltante: {perm.value}',
                        'blocked': True,
                    }
            
            # Ejecutar en sandbox con el primer permiso
            first_perm = next(iter(tool.required_permissions))
            result = self.sandbox.execute(tool.func, first_perm, *args, **kwargs)
            
            if not result.success:
                return {'success': False, 'error': result.error, 'blocked': result.blocked}
            
            return {'success': True, 'result': result.result}
        else:
            # Sin sandbox, ejecutar directamente
            try:
                result = tool.func(*args, **kwargs)
                return {'success': True, 'result': result}
            except Exception as e:
                return {'success': False, 'error': str(e)}
    
    def __repr__(self):
        return f"ToolRegistry({len(self.tools)} herramientas)"


if __name__ == "__main__":
    print("🧪 PROBANDO TOOL REGISTRY")
    print("="*50)
    
    from security.sandbox import Sandbox
    
    sandbox = Sandbox(strict=True)
    registry = ToolRegistry(sandbox=sandbox)
    
    # Registrar herramientas
    @registry.register(
        name="calculadora",
        description="Suma, resta, multiplica, divide",
        permissions={Permission.EXECUTE_MATH},
    )
    def calculadora(a, b, op='+'):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return a / b if b != 0 else "Error: división por cero"
        return "Operación no válida"
    
    @registry.register(
        name="fecha",
        description="Fecha actual",
        permissions={Permission.READ_SYSTEM_INFO},
    )
    def fecha():
        import datetime
        return datetime.datetime.now().strftime('%d/%m/%Y')
    
    # Listar
    print("\n📋 Herramientas registradas:")
    for t in registry.list_tools():
        print(f"   {t['name']}: {t['description']}")
    
    # Ejecutar sin permiso
    print("\n1. Ejecutar calculadora SIN permiso:")
    r = registry.execute('calculadora', 5, 3, '+')
    print(f"   {r}")
    
    # Otorgar permiso
    sandbox.permissions.add(Permission.EXECUTE_MATH)
    print("\n2. Ejecutar calculadora CON permiso:")
    r = registry.execute('calculadora', 5, 3, '+')
    print(f"   {r}")
    
    # Ejecutar fecha
    sandbox.permissions.add(Permission.READ_SYSTEM_INFO)
    print("\n3. Ejecutar fecha:")
    r = registry.execute('fecha')
    print(f"   {r}")
    
    print("\n✅ TOOL REGISTRY FUNCIONANDO")
