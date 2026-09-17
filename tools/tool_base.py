"""
Clase base para herramientas (plugins)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Set, Any
from security.permissions import Permission


class Tool:
    """
    Clase base para herramientas (plugins).
    
    Los plugins heredan de esta clase y sobrescriben:
    - `name` (str)
    - `description` (str)
    - `permissions` (Set[Permission])
    - `can_handle(input_text) -> bool`
    - `execute(input_text) -> dict`
    """
    
    name: str = "tool"
    description: str = "Herramienta genérica"
    permissions: Set[Permission] = set()
    dangerous: bool = False
    
    def can_handle(self, input_text: str) -> bool:
        """Determina si puede manejar la entrada"""
        raise NotImplementedError("Debe implementar can_handle()")
    
    def execute(self, input_text: str) -> Any:
        """Ejecuta la herramienta sobre la entrada"""
        raise NotImplementedError("Debe implementar execute()")
    
    def __repr__(self):
        return f"Tool({self.name}, perms={len(self.permissions)})"
