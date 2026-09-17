"""
Plugin: Reloj
Detecta preguntas sobre hora y fecha.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from datetime import datetime
from tools.tool_base import Tool
from security.permissions import Permission


class ClockTool(Tool):
    name = "reloj"
    description = "Devuelve la hora o fecha actual"
    permissions = {Permission.READ_SYSTEM_INFO}
    dangerous = False
    
    HORA_PATTERN = re.compile(r'(?:qu[eé]\s+hora|hora\s+es|dime\s+la\s+hora)', re.IGNORECASE)
    FECHA_PATTERN = re.compile(r'(?:qu[eé]\s+fecha|fecha\s+es|dime\s+la\s+fecha|qu[eé]\s+d[ií]a)', re.IGNORECASE)
    
    def can_handle(self, input_text: str) -> bool:
        return bool(self.HORA_PATTERN.search(input_text) or self.FECHA_PATTERN.search(input_text))
    
    def execute(self, input_text: str) -> dict:
        if self.HORA_PATTERN.search(input_text):
            return {
                'success': True,
                'result': datetime.now().strftime('%H:%M:%S'),
                'type': 'hora',
            }
        if self.FECHA_PATTERN.search(input_text):
            return {
                'success': True,
                'result': datetime.now().strftime('%d/%m/%Y'),
                'type': 'fecha',
            }
        return {'success': False, 'error': 'No aplicable'}


tool = ClockTool()
