"""
Tool Registry v2 con descubrimiento de plugins
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Any
from tools.tool_base import Tool
from tools.loader import discover_plugins
from security.sandbox import Sandbox


class ToolRegistryV2:
    """
    Registro de herramientas basado en plugins.
    
    - Descubre automáticamente plugins en `tools/plugins/`
    - Filtra por permisos del sandbox
    - Ejecuta según `can_handle`
    """
    
    def __init__(self, sandbox: Sandbox = None, auto_load: bool = True):
        self.sandbox = sandbox
        self.plugins: Dict[str, Tool] = {}
        
        if auto_load:
            self.load_plugins()
    
    def load_plugins(self):
        """Carga los plugins disponibles"""
        self.plugins = discover_plugins()
    
    def register(self, tool: Tool):
        """Registra manualmente un plugin"""
        self.plugins[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self.plugins.get(name)
    
    def list_tools(self) -> List[dict]:
        return [
            {
                'name': t.name,
                'description': t.description,
                'permissions': [p.value for p in t.permissions],
                'dangerous': t.dangerous,
            }
            for t in self.plugins.values()
        ]
    
    def find_handler(self, input_text: str) -> Optional[Tool]:
        """
        Encuentra el primer plugin que pueda manejar la entrada.
        Respeta el orden de las herramientas (por nombre).
        """
        for name in sorted(self.plugins.keys()):
            tool = self.plugins[name]
            try:
                if tool.can_handle(input_text):
                    return tool
            except Exception:
                continue
        return None
    
    def execute(self, name: str, input_text: str) -> dict:
        """Ejecuta un plugin por nombre"""
        tool = self.plugins.get(name)
        if not tool:
            return {'success': False, 'error': f'Plugin no encontrado: {name}'}
        
        # Verificar permisos
        if self.sandbox:
            for perm in tool.permissions:
                if not self.sandbox.permissions.has(perm):
                    return {
                        'success': False,
                        'error': f'Permiso faltante: {perm.value}',
                        'blocked': True,
                    }
        
        # Ejecutar
        try:
            return tool.execute(input_text)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def execute_by_input(self, input_text: str) -> dict:
        """Encuentra el handler y ejecuta"""
        tool = self.find_handler(input_text)
        if tool is None:
            return {'success': False, 'error': 'No hay handler disponible'}
        return self.execute(tool.name, input_text)
    
    def __repr__(self):
        return f"ToolRegistryV2({len(self.plugins)} plugins)"


if __name__ == "__main__":
    print("🧪 PROBANDO TOOL REGISTRY V2")
    print("="*50)
    
    from security.sandbox import Sandbox
    from security.permissions import Permission, PERMISSIONS_STANDARD
    
    sandbox = Sandbox(permissions=PERMISSIONS_STANDARD.copy(), strict=False)
    registry = ToolRegistryV2(sandbox=sandbox)
    
    print(f"\n📦 Plugins registrados: {len(registry.plugins)}")
    for name, tool in registry.plugins.items():
        print(f"   ✅ {name}: {tool.description}")
    
    # Probar
    print(f"\n🔍 Probando ejecución:")
    pruebas = [
        "¿Cuánto es 5 + 3?",
        "¿Qué hora es?",
        "hola",
        "adiós",
        "cuenta las palabras de 'hola mundo cruel'",
        "algo sin sentido",
    ]
    
    for p in pruebas:
        print(f"\n   Entrada: {p}")
        handler = registry.find_handler(p)
        if handler:
            print(f"   Handler: {handler.name}")
            result = registry.execute_by_input(p)
            print(f"   Resultado: {result}")
        else:
            print(f"   ❌ Sin handler")
    
    print("\n✅ TOOL REGISTRY V2 FUNCIONANDO")
