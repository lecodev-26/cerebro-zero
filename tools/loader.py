"""
Cargador automático de plugins de herramientas.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import pkgutil
from typing import Dict, List
from tools.tool_base import Tool


def discover_plugins(package_name: str = "tools.plugins") -> Dict[str, Tool]:
    """
    Descubre todos los plugins en el paquete dado.
    Busca la variable `tool` en cada módulo.
    """
    plugins = {}
    
    try:
        package = importlib.import_module(package_name)
        package_path = package.__path__
    except ImportError as e:
        print(f"⚠️ No se pudo importar {package_name}: {e}")
        return plugins
    
    for finder, name, ispkg in pkgutil.iter_modules(package_path):
        if name.startswith('_'):
            continue
        
        full_name = f"{package_name}.{name}"
        try:
            module = importlib.import_module(full_name)
            
            # Buscar variable `tool` (instancia)
            if hasattr(module, 'tool'):
                tool_instance = module.tool
                if isinstance(tool_instance, Tool):
                    plugins[tool_instance.name] = tool_instance
        except Exception as e:
            print(f"⚠️ Error cargando plugin {full_name}: {e}")
    
    return plugins


def list_plugins() -> List[str]:
    """Lista los nombres de plugins disponibles"""
    return sorted(discover_plugins().keys())


if __name__ == "__main__":
    print("🧪 PROBANDO LOADER DE PLUGINS")
    print("="*50)
    
    plugins = discover_plugins()
    
    print(f"\n📦 Plugins descubiertos: {len(plugins)}")
    for name, tool in plugins.items():
        print(f"   ✅ {name}: {tool.description}")
    
    print("\n📋 Nombres:", list_plugins())
    
    print("\n✅ LOADER FUNCIONANDO")
