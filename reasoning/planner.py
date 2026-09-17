"""
Planner serio para Cerebro Zero
Descompone problemas en subproblemas y los resuelve uno a uno.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from dataclasses import dataclass, field
from typing import Any, Optional, List, Callable


@dataclass
class Step:
    """Un paso del plan"""
    kind: str               # 'math', 'lookup', 'definition', 'tool', 'unknown'
    description: str        # descripción legible
    execute: Callable = None  # función que ejecuta el paso
    result: Any = None      # resultado tras ejecutar
    error: Optional[str] = None
    
    def run(self) -> Any:
        if self.execute is None:
            self.error = "Paso sin función"
            return None
        try:
            self.result = self.execute()
            return self.result
        except Exception as e:
            self.error = str(e)
            return None


@dataclass
class Plan:
    """Plan completo para resolver una entrada"""
    input: str
    steps: List[Step] = field(default_factory=list)
    final_answer: Any = None
    error: Optional[str] = None
    
    def run(self) -> Any:
        """Ejecuta todos los pasos en orden"""
        for step in self.steps:
            step.run()
        # La respuesta es el último resultado válido
        for step in reversed(self.steps):
            if step.result is not None:
                self.final_answer = step.result
                return self.final_answer
        return None
    
    def is_empty(self) -> bool:
        return len(self.steps) == 0


class Planner:
    """
    Planner que descompone la entrada en pasos.
    
    Estrategia:
    1. Si hay operación matemática → paso 'math'
    2. Si hay pregunta "¿qué es X?" → paso 'definition'
    3. Si hay pregunta "¿cuándo/hora?" → paso 'tool' con la herramienta
    4. Si no → paso 'unknown' (no resuelve)
    """
    
    # Patrones
    MATH_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)')
    WHAT_IS_PATTERN = re.compile(r'(?:qu[eé]\s+es|qu[eé]\s+significa|define)\s+(.+)', re.IGNORECASE)
    HORA_PATTERN = re.compile(r'(?:qu[eé]\s+hora|hora\s+es)', re.IGNORECASE)
    FECHA_PATTERN = re.compile(r'(?:qu[eé]\s+fecha|fecha\s+es)', re.IGNORECASE)
    
    def __init__(self):
        self.knowledge_base = {}  # Se puede llenar con conocimientos
    
    def set_knowledge(self, base: dict):
        """Configura la base de conocimiento para definiciones"""
        self.knowledge_base = {k.lower(): v for k, v in base.items()}
    
    def planificar(self, pregunta: str) -> Plan:
        """Crea un plan para responder la pregunta"""
        plan = Plan(input=pregunta)
        pregunta_lower = pregunta.lower().strip()
        
        # 1. ¿Es una operación matemática?
        math_match = self.MATH_PATTERN.search(pregunta)
        if math_match:
            a, op, b = math_match.groups()
            step = Step(
                kind='math',
                description=f"Calcular {a} {op} {b}",
                execute=self._make_math_executor(float(a), op, float(b)),
            )
            plan.steps.append(step)
            return plan
        
        # 2. ¿Es una pregunta "¿qué es X?"
        what_match = self.WHAT_IS_PATTERN.search(pregunta)
        if what_match:
            subject = what_match.group(1).strip().rstrip('?').strip()
            step = Step(
                kind='definition',
                description=f"Buscar definición de '{subject}'",
                execute=self._make_definition_executor(subject),
            )
            plan.steps.append(step)
            return plan
        
        # 3. ¿Pregunta por hora?
        if self.HORA_PATTERN.search(pregunta):
            step = Step(
                kind='tool',
                description="Obtener hora actual",
                execute=self._get_time,
            )
            plan.steps.append(step)
            return plan
        
        # 4. ¿Pregunta por fecha?
        if self.FECHA_PATTERN.search(pregunta):
            step = Step(
                kind='tool',
                description="Obtener fecha actual",
                execute=self._get_date,
            )
            plan.steps.append(step)
            return plan
        
        # 5. No sé cómo resolverlo
        step = Step(
            kind='unknown',
            description="No hay estrategia conocida para esta entrada",
            execute=None,
        )
        plan.steps.append(step)
        return plan
    
    # ============================================
    # EXECUTORES
    # ============================================
    
    def _make_math_executor(self, a: float, op: str, b: float) -> Callable:
        def execute():
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '/':
                if b == 0:
                    raise ValueError("División por cero")
                return a / b
            raise ValueError(f"Operador desconocido: {op}")
        return execute
    
    def _make_definition_executor(self, subject: str) -> Callable:
        def execute():
            subject_lower = subject.lower()
            if subject_lower in self.knowledge_base:
                return self.knowledge_base[subject_lower]
            # Búsqueda parcial
            for key, value in self.knowledge_base.items():
                if key in subject_lower or subject_lower in key:
                    return value
            return None  # No sabe
        return execute
    
    def _get_time(self):
        from datetime import datetime
        return datetime.now().strftime('%H:%M:%S')
    
    def _get_date(self):
        from datetime import datetime
        return datetime.now().strftime('%d/%m/%Y')


class Verifier:
    """
    Verificador que comprueba si una respuesta es fiable.
    
    Reglas:
    - Si la respuesta es un número y la pregunta era matemática → OK
    - Si la respuesta es None → error
    - Si la respuesta contiene "no sé" o "no tengo" → baja confianza
    """
    
    def verificar(self, respuesta: Any, plan: Plan) -> tuple:
        """
        Devuelve (ok, confianza, razón)
        """
        # 1. Respuesta vacía
        if respuesta is None:
            return False, 0.0, "Respuesta vacía"
        
        # 2. Respuesta en texto plano
        respuesta_str = str(respuesta).strip()
        if not respuesta_str:
            return False, 0.0, "Respuesta vacía"
        
        # 3. Respuestas "no sé"
        negative_phrases = [
            'no sé', 'no se', 'no tengo', 'no entiendo',
            'no hay', 'no puedo', 'sin información',
        ]
        respuesta_lower = respuesta_str.lower()
        for phrase in negative_phrases:
            if phrase in respuesta_lower:
                return False, 0.3, f"Respuesta negativa: '{phrase}'"
        
        # 4. Verificar según el tipo de paso
        if plan.steps:
            last_step = plan.steps[-1]
            if last_step.kind == 'math':
                # El resultado debe ser numérico
                try:
                    float(respuesta_str)
                    return True, 0.95, "Cálculo matemático verificado"
                except ValueError:
                    return False, 0.4, "Resultado no numérico"
            
            if last_step.kind == 'definition':
                # La respuesta debe estar en el knowledge base
                if last_step.result is None:
                    return False, 0.3, "Definición no encontrada"
                return True, 0.85, "Definición encontrada"
            
            if last_step.kind == 'tool':
                # Herramienta ejecutada → confiar
                return True, 0.9, "Herramienta ejecutada"
            
            if last_step.kind == 'unknown':
                return False, 0.2, "Sin estrategia conocida"
        
        # 5. Respuesta por defecto
        return True, 0.5, "Respuesta sin verificar"


class Razonamiento:
    """
    Interfaz unificada que combina Planner + Verifier.
    
    Compatible con el pipeline existente.
    """
    
    def __init__(self, verbose: bool = True):
        self.planner = Planner()
        self.verifier = Verifier()
        self.verbose = verbose
        self.historial = []
    
    def razonar(self, pregunta: str) -> list:
        """
        Devuelve una lista con la respuesta (o lista vacía si no sabe).
        Compatible con `step_reasoning` del pipeline.
        """
        plan = self.planner.planificar(pregunta)
        
        if self.verbose:
            print(f"🧠 Razonando sobre: {pregunta}")
            print(f"📋 Plan: {len(plan.steps)} paso(s)")
            for s in plan.steps:
                print(f"   - [{s.kind}] {s.description}")
        
        if plan.is_empty() or plan.steps[0].kind == 'unknown':
            if self.verbose:
                print("   ❌ Sin estrategia conocida")
            return []
        
        # Ejecutar
        resultado = plan.run()
        
        # Verificar
        ok, confianza, razon = self.verifier.verificar(resultado, plan)
        
        if self.verbose:
            print(f"🔍 Verificación: ok={ok}, confianza={confianza:.2f}, razón={razon}")
        
        self.historial.append({
            'pregunta': pregunta,
            'plan': plan,
            'resultado': resultado,
            'ok': ok,
            'confianza': confianza,
        })
        
        if not ok:
            return []
        
        return [str(resultado)]
    
    def set_knowledge(self, base: dict):
        """Configura la base de conocimiento"""
        self.planner.set_knowledge(base)


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO PLANNER + VERIFIER")
    print("="*50)
    
    r = Razonamiento(verbose=True)
    r.set_knowledge({
        'ia': 'Inteligencia Artificial: capacidad de máquinas para imitar funciones humanas.',
        'python': 'Python es un lenguaje de programación de alto nivel.',
        'cerebro zero': 'Cerebro Zero es un agente híbrido neuronal construido desde cero.',
    })
    
    entradas = [
        "¿Cuánto es 5 + 3?",
        "¿Cuánto es 10 * 7?",
        "¿Cuánto es 15 / 3?",
        "¿Qué es la IA?",
        "¿Qué es Python?",
        "¿Qué es Cerebro Zero?",
        "¿Qué hora es?",
        "¿Qué fecha es hoy?",
        "algo que no sé",
    ]
    
    for e in entradas:
        print(f"\n📝 Entrada: {e}")
        resultado = r.razonar(e)
        print(f"   Resultado: {resultado}")
    
    print("\n✅ PLANNER + VERIFIER FUNCIONANDO")
