"""
Plugin: Saludos y despedidas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from tools.tool_base import Tool
from security.permissions import Permission


class GreetingTool(Tool):
    name = "saludo"
    description = "Responde a saludos y despedidas"
    permissions = {Permission.READ_MEMORY}
    dangerous = False
    
    SALUDOS = re.compile(r'\b(hola|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|hey|qué\s+tal)\b', re.IGNORECASE)
    DESPEDIDAS = re.compile(r'\b(adi[oó]s|hasta\s+luego|chao|nos\s+vemos|bye)\b', re.IGNORECASE)
    
    def can_handle(self, input_text: str) -> bool:
        return bool(self.SALUDOS.search(input_text) or self.DESPEDIDAS.search(input_text))
    
    def execute(self, input_text: str) -> dict:
        if self.SALUDOS.search(input_text):
            return {
                'success': True,
                'result': "¡Hola! Soy Cerebro Zero. ¿En qué puedo ayudarte?",
                'type': 'saludo',
            }
        if self.DESPEDIDAS.search(input_text):
            return {
                'success': True,
                'result': "¡Hasta luego! Ha sido un placer.",
                'type': 'despedida',
            }
        return {'success': False, 'error': 'No aplicable'}


tool = GreetingTool()
