"""
Plugin: Contador de palabras
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from tools.tool_base import Tool
from security.permissions import Permission


class WordCounterTool(Tool):
    name = "contador"
    description = "Cuenta las palabras de un texto"
    permissions = {Permission.READ_MEMORY}
    dangerous = False
    
    PATTERN = re.compile(r'cuenta\s+(?:las\s+)?palabras\s+(?:de\s+)?["\']?(.+)["\']?', re.IGNORECASE)
    
    def can_handle(self, input_text: str) -> bool:
        return bool(self.PATTERN.search(input_text))
    
    def execute(self, input_text: str) -> dict:
        match = self.PATTERN.search(input_text)
        if not match:
            return {'success': False, 'error': 'No aplicable'}
        
        texto = match.group(1).strip()
        palabras = texto.split()
        
        return {
            'success': True,
            'result': f"El texto tiene {len(palabras)} palabra(s)",
            'value': len(palabras),
        }


tool = WordCounterTool()
