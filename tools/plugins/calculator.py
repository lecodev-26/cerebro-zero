"""
Plugin: Calculadora
Detecta operaciones matemáticas y las resuelve.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from tools.tool_base import Tool
from security.permissions import Permission


class CalculatorTool(Tool):
    name = "calculadora"
    description = "Resuelve operaciones matemáticas (+, -, *, /)"
    permissions = {Permission.EXECUTE_MATH}
    dangerous = False
    
    PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)')
    
    def can_handle(self, input_text: str) -> bool:
        return bool(self.PATTERN.search(input_text))
    
    def execute(self, input_text: str) -> dict:
        match = self.PATTERN.search(input_text)
        if not match:
            return {'success': False, 'error': 'No hay operación'}
        
        a, op, b = match.groups()
        a, b = float(a), float(b)
        
        if op == '+': result = a + b
        elif op == '-': result = a - b
        elif op == '*': result = a * b
        elif op == '/':
            if b == 0:
                return {'success': False, 'error': 'División por cero'}
            result = a / b
        else:
            return {'success': False, 'error': f'Operador desconocido: {op}'}
        
        return {
            'success': True,
            'result': f"{a} {op} {b} = {result}",
            'value': result,
        }


# Instancia para el registro
tool = CalculatorTool()
