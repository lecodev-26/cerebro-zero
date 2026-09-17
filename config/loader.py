"""
Cargador de configuración YAML
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import copy
from typing import Any, Dict, Optional


class Config:
    """
    Configuración cargada desde YAML.
    
    Uso:
        config = Config.load()
        lr = config.get('training.learning_rate', default=0.01)
        model_type = config.get('model.type', default='transformer')
    """
    
    def __init__(self, data: dict = None):
        self.data = data or {}
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'Config':
        """
        Carga la configuración desde un archivo YAML.
        
        Args:
            path: ruta al archivo YAML. Si es None, busca 'config/default.yaml'.
        
        Returns:
            Config con la configuración cargada.
        """
        if path is None:
            # Buscar en ubicaciones comunes
            for candidate in ['config/default.yaml', 'default.yaml']:
                if os.path.exists(candidate):
                    path = candidate
                    break
        
        if path is None or not os.path.exists(path):
            print(f"⚠️ No se encontró {path}. Usando config vacía.")
            return cls({})
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(data)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtiene un valor por ruta de clave (ej: 'training.learning_rate').
        
        Args:
            key_path: ruta de la clave separada por puntos
            default: valor por defecto si no existe
        
        Returns:
            El valor encontrado o el default.
        """
        keys = key_path.split('.')
        value = self.data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Establece un valor por ruta de clave.
        """
        keys = key_path.split('.')
        target = self.data
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
    
    def section(self, name: str) -> dict:
        """Devuelve una sección completa como dict"""
        return self.data.get(name, {})
    
    def to_dict(self) -> dict:
        """Devuelve una copia PROFUNDA de la configuración"""
        return copy.deepcopy(self.data)
    
    def __repr__(self):
        return f"Config({len(self.data)} secciones)"


# ============================================
# HELPER GLOBAL
# ============================================

_GLOBAL_CONFIG = None


def get_config(path: Optional[str] = None) -> Config:
    """
    Devuelve la configuración global (singleton).
    """
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = Config.load(path)
    return _GLOBAL_CONFIG


def reset_config():
    """Resetea la configuración global (útil para tests)"""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO CONFIG LOADER")
    print("="*60)
    
    # Cargar
    config = Config.load('config/default.yaml')
    print(f"\n✅ Config cargada: {config}")
    
    # Test get con ruta
    print("\n📖 Valores:")
    print(f"   model.type = {config.get('model.type')}")
    print(f"   model.d_model = {config.get('model.d_model')}")
    print(f"   model.num_layers = {config.get('model.num_layers')}")
    print(f"   training.learning_rate = {config.get('training.learning_rate')}")
    print(f"   training.batch_size = {config.get('training.batch_size')}")
    print(f"   memory.short_term.max_size = {config.get('memory.short_term.max_size')}")
    print(f"   security.sandbox.strict = {config.get('security.sandbox.strict')}")
    print(f"   logging.level = {config.get('logging.level')}")
    
    # Test default
    print("\n📖 Valores inexistentes (con default):")
    print(f"   model.nonexistent = {config.get('model.nonexistent', 'DEFAULT')}")
    print(f"   training.xyz.abc = {config.get('training.xyz.abc', 42)}")
    
    # Test section
    print(f"\n📖 Sección 'model':")
    model_section = config.section('model')
    for key, value in model_section.items():
        print(f"   {key}: {value}")
    
    # Test get_config global
    print("\n📖 Config global (singleton):")
    global_config = get_config()
    print(f"   {global_config}")
    print(f"   Mismo objeto: {global_config is config}")
    
    print("\n✅ CONFIG LOADER FUNCIONANDO")
