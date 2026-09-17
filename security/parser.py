"""
Parser de comandos con detección de intenciones peligrosas
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
    intent: str  # 'math', 'memory', 'filesystem', 'network', 'unknown'
    entities: dict  # entidades extraídas (números, nombres, etc.)
    dangerous: bool
    danger_reason: Optional[str] = None


class Parser:
    """Parser de comandos con detección de peligros"""
    
    # Patrones de intención
    INTENTS = {
        'math': [
            r'\d+\s*[+\-*/]\s*\d+',
            r'(suma|resta|multiplica|divide|calcula)',
        ],
        'memory': [
            r'(recuerda|memoriza|aprende|guarda)',
            r'(olvida|borra memoria)',
        ],
        'filesystem': [
            r'(archivo|directorio|carpeta|fichero)',
            r'(lee|escribe|crea|borra|elimina)\s+(archivo|carpeta)',
        ],
        'network': [
            r'(http|https|url|descarga|sube)',
            r'(clima|tiempo|noticias|busca en internet)',
        ],
        'system': [
            r'(sistema|proceso|programa|ejecuta)',
            r'(instala|desinstala|actualiza)',
        ],
    }
    
    # Patrones peligrosos (señales de riesgo)
    DANGEROUS_PATTERNS = [
        (r'rm\s+-rf', 'Intento de borrar recursivamente'),
        (r'format\s+[a-z]:', 'Intento de formatear disco'),
        (r'del\s+/[fqs]', 'Intento de borrar forzado'),
        (r'eval\s*\(', 'Uso de eval()'),
        (r'exec\s*\(', 'Uso de exec()'),
        (r'__import__', 'Importación dinámica'),
        (r'os\.system', 'Llamada a sistema'),
        (r'subprocess\.', 'Uso de subprocess'),
        (r'open\s*\(.*[\'"]w[\'"]', 'Apertura de archivo en escritura'),
        (r'sudo\s+', 'Uso de sudo'),
        (r'chmod\s+777', 'Cambio de permisos inseguro'),
        (r'curl\s+.*\|\s*sh', 'Piping a shell'),
        (r'wget\s+.*\|\s*sh', 'Piping a shell'),
        (r';\s*rm\s+', 'Inyección de comando'),
        (r'`.*`', 'Ejecución por backticks'),
        (r'\$\(.*\)', 'Sustitución de comandos'),
    ]
    
    # Números
    NUMBER_PATTERN = r'[-+]?\d*\.?\d+'
    
    def parse(self, input_text: str) -> ParsedCommand:
        """Analiza una entrada y devuelve un ParsedCommand"""
        raw = input_text
        normalized = input_text.strip().lower()
        
        # 1. Detectar intención
        intent = self._detect_intent(normalized)
        
        # 2. Extraer entidades
        entities = self._extract_entities(normalized, intent)
        
        # 3. Detectar peligros
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
        """Detecta la intención del comando"""
        for intent, patterns in self.INTENTS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        return 'unknown'
    
    def _extract_entities(self, text: str, intent: str) -> dict:
        """Extrae entidades según la intención"""
        entities = {}
        
        if intent == 'math':
            numeros = re.findall(self.NUMBER_PATTERN, text)
            entities['numbers'] = [float(n) for n in numeros]
            
            # Detectar operador
            if '+' in text or 'suma' in text:
                entities['operator'] = '+'
            elif '-' in text or 'resta' in text:
                entities['operator'] = '-'
            elif '*' in text or 'multiplica' in text:
                entities['operator'] = '*'
            elif '/' in text or 'divide' in text:
                entities['operator'] = '/'
        
        elif intent == 'network':
            # Extraer URL si hay
            urls = re.findall(r'https?://[^\s]+', text)
            if urls:
                entities['urls'] = urls
        
        return entities
    
    def _detect_dangerous(self, text: str) -> tuple:
        """Detecta si el texto contiene patrones peligrosos"""
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, reason
        return False, None
    
    def __repr__(self):
        return f"Parser({len(self.INTENTS)} intenciones, {len(self.DANGEROUS_PATTERNS)} patrones peligrosos)"


if __name__ == "__main__":
    print("🧪 PROBANDO PARSER")
    print("="*50)
    
    parser = Parser()
    
    casos = [
        "¿Cuánto es 5 + 3?",
        "suma 10 y 20",
        "recuerda que me llamo Manuel",
        "rm -rf /",
        "eval('print(1)')",
        "descarga https://ejemplo.com/archivo.txt",
        "¿Qué hora es?",
        "os.system('ls')",
    ]
    
    for caso in casos:
        print(f"\n📝 Entrada: {caso}")
        parsed = parser.parse(caso)
        print(f"   Intención: {parsed.intent}")
        print(f"   Entidades: {parsed.entities}")
        print(f"   ¿Peligroso?: {parsed.dangerous}")
        if parsed.dangerous:
            print(f"   ⚠️ Razón: {parsed.danger_reason}")
    
    print("\n✅ PARSER FUNCIONANDO")
