"""
Parser de comandos con detección de intenciones y peligros
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedCommand:
    """Resultado del parsing"""
    raw: str
    normalized: str
    intent: str
    entities: dict
    dangerous: bool
    danger_reason: Optional[str] = None


class Parser:
    """Parser de comandos"""
    
    # Patrones de intención
    INTENTS = {
        'math': [
            r'\d+\s*[+\-*/]\s*\d+',
            r'(suma|resta|multiplica|divide|calcula)',
        ],
        'memory_store': [
            r'recuerda\s+\w+\s*=\s*\S+',
            r'memoriza\s+\w+\s*=\s*\S+',
            r'guarda\s+\w+\s*=\s*\S+',
        ],
        'memory_recall': [
            r'recordar\s+\w+',
            r'qué\s+es\s+\w+',
            r'qu[eé]\s+sabes\s+de\s+\w+',
        ],
        'learning': [
            r'aprende\s+\w+',
        ],
        'greeting': [
            r'^(hola|buenos|buenas|hey|qué\s+tal|saludos|qué\s+pasa|qué\s+hay)\b',
        ],
        'farewell': [
            r'^(adi[oó]s|hasta\s+luego|chao|nos\s+vemos|bye)\b',
        ],
        'time': [
            r'(qu[eé]\s+hora|dime\s+la\s+hora|hora\s+es|hora\s+actual)',
            r'^hora$',
        ],
        'date': [
            r'(qu[eé]\s+fecha|dime\s+la\s+fecha|fecha\s+es|qu[eé]\s+d[ií]a)',
            r'^fecha$',
            r'^hoy\??$',
        ],
    }
    
    # Patrones peligrosos
    DANGEROUS_PATTERNS = [
        (r'rm\s+-rf', 'Intento de borrar recursivamente'),
        (r'format\s+[a-z]:', 'Intento de formatear disco'),
        (r'del\s+/[fqs]', 'Intento de borrar forzado'),
        (r'eval\s*\(', 'Uso de eval()'),
        (r'exec\s*\(', 'Uso de exec()'),
        (r'__import__', 'Importación dinámica'),
        (r'os\.system', 'Llamada a sistema'),
        (r'subprocess\.', 'Uso de subprocess'),
        (r'sudo\s+', 'Uso de sudo'),
        (r'chmod\s+777', 'Cambio de permisos inseguro'),
        (r'curl\s+.*\|\s*sh', 'Piping a shell'),
        (r'wget\s+.*\|\s*sh', 'Piping a shell'),
        (r';\s*rm\s+', 'Inyección de comando'),
        (r'`.*`', 'Ejecución por backticks'),
        (r'\$\(.*\)', 'Sustitución de comandos'),
    ]
    
    NUMBER_PATTERN = r'[-+]?\d*\.?\d+'
    
    def parse(self, input_text: str) -> ParsedCommand:
        """Analiza una entrada y devuelve un ParsedCommand"""
        raw = input_text
        normalized = input_text.strip().lower()
        
        intent = self._detect_intent(normalized)
        entities = self._extract_entities(normalized, intent, raw)
        dangerous, reason = self._detect_dangerous(raw)
        
        return ParsedCommand(
            raw=raw,
            normalized=normalized,
            intent=intent,
            entities=entities,
            dangerous=dangerous,
            danger_reason=reason,
        )
    
    def _detect_intent(self, text: str) -> str:
        """Detecta la intención del comando (orden de prioridad)"""
        # Orden específico: memory_store > memory_recall > learning > time > date
        # > greeting > farewell > math > unknown
        
        priority_order = [
            'memory_store', 'learning', 'memory_recall',
            'time', 'date',
            'greeting', 'farewell',
            'math',
        ]
        
        for intent in priority_order:
            for pattern in self.INTENTS[intent]:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        
        return 'unknown'
    
    def _extract_entities(self, text: str, intent: str, raw: str) -> dict:
        """Extrae entidades según la intención"""
        entities = {}
        
        if intent == 'math':
            numeros = re.findall(self.NUMBER_PATTERN, text)
            entities['numbers'] = [float(n) for n in numeros]
            
            if '+' in text or 'suma' in text:
                entities['operator'] = '+'
            elif '-' in text or 'resta' in text:
                entities['operator'] = '-'
            elif '*' in text or 'multiplica' in text:
                entities['operator'] = '*'
            elif '/' in text or 'divide' in text:
                entities['operator'] = '/'
        
        elif intent == 'memory_store':
            # "recuerda clave = valor"
            match = re.search(r'(?:recuerda|memoriza|guarda)\s+(\S+)\s*=\s*(.+)', raw, re.IGNORECASE)
            if match:
                entities['key'] = match.group(1).strip()
                entities['value'] = match.group(2).strip()
        
        elif intent == 'memory_recall':
            # "recordar clave"
            match = re.search(r'recordar\s+(\S+)', raw, re.IGNORECASE)
            if match:
                entities['key'] = match.group(1).strip()
            else:
                # "qué es X" → clave
                match = re.search(r'(?:qu[eé]\s+es|qu[eé]\s+sabes\s+de)\s+(\S+)', raw, re.IGNORECASE)
                if match:
                    entities['key'] = match.group(1).strip()
        
        elif intent == 'learning':
            match = re.search(r'aprende\s+(.+)', raw, re.IGNORECASE)
            if match:
                entities['text'] = match.group(1).strip()
        
        return entities
    
    def _detect_dangerous(self, text: str) -> tuple:
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, reason
        return False, None


if __name__ == "__main__":
    print("🧪 PROBANDO PARSER V3.0")
    print("="*60)
    
    parser = Parser()
    
    casos = [
        "¿Cuánto es 5 + 3?",
        "recuerda color = azul",
        "recordar color",
        "aprende a sumar",
        "¿Qué hora es?",
        "dime la hora",
        "¿Qué fecha es hoy?",
        "hola",
        "adiós",
        "rm -rf /",
        "eval('print(1)')",
        "qué es la ia",
    ]
    
    for caso in casos:
        parsed = parser.parse(caso)
        print(f"\n📝 '{caso}'")
        print(f"   Intent: {parsed.intent}")
        print(f"   Entities: {parsed.entities}")
        if parsed.dangerous:
            print(f"   ⚠️ PELIGROSO: {parsed.danger_reason}")
    
    print("\n✅ PARSER V3.0 FUNCIONANDO")
