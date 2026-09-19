"""
Pipeline del Cerebro Central con memoria y aprendizaje
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict


@dataclass
class Context:
    """Contexto que fluye por el pipeline"""
    input: str = ""
    parsed: Any = None
    allowed: bool = False
    block_reason: Optional[str] = None
    tool_result: Optional[Any] = None
    memory_results: List[dict] = field(default_factory=list)
    exact_match: Optional[str] = None
    memory_stored: bool = False
    reasoning_steps: List[Any] = field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0
    output: Optional[str] = None
    error: Optional[str] = None
    learned: bool = False
    
    def to_dict(self) -> dict:
        return {
            'input': self.input,
            'intent': self.parsed.intent if self.parsed else None,
            'allowed': self.allowed,
            'output': self.output,
            'learned': self.learned,
            'memory_stored': self.memory_stored,
        }


# ============================================
# PASOS DEL PIPELINE
# ============================================

def step_parse(ctx: Context, parser) -> Context:
    """Paso 1: Parsear la entrada"""
    try:
        ctx.parsed = parser.parse(ctx.input)
    except Exception as e:
        ctx.error = f"Error al parsear: {e}"
    return ctx


def step_sandbox_check(ctx: Context, sandbox) -> Context:
    """Paso 2: Verificar seguridad"""
    if ctx.parsed is None:
        ctx.error = "No hay parsed"
        return ctx
    
    if ctx.parsed.dangerous:
        ctx.allowed = False
        ctx.block_reason = ctx.parsed.danger_reason
        return ctx
    
    ctx.allowed = True
    return ctx


def step_memory_store(ctx: Context, agent) -> Context:
    """Paso 3a: Almacenar en memoria si es un comando 'recuerda X = Y'"""
    if not ctx.allowed or ctx.parsed is None:
        return ctx
    
    if ctx.parsed.intent == 'memory_store':
        key = ctx.parsed.entities.get('key')
        value = ctx.parsed.entities.get('value')
        if key and value is not None:
            try:
                agent.memory.add_knowledge(key, value, {'tipo': 'memoria'})
                ctx.memory_stored = True
                ctx.output = f"✅ Guardado: {key} = {value}"
            except Exception as e:
                ctx.error = f"Error guardando: {e}"
    
    return ctx


def step_memory_recall(ctx: Context, agent) -> Context:
    """Paso 3b: Recuperar de memoria si es 'recordar X'"""
    if not ctx.allowed or ctx.parsed is None:
        return ctx
    
    if ctx.output is not None:  # Ya hay output
        return ctx
    
    if ctx.parsed.intent in ('memory_recall',):
        key = ctx.parsed.entities.get('key')
        if key:
            try:
                # Buscar en memoria de largo plazo
                results = agent.memory.long_term.search(key)
                if results:
                    value = results[0].get('value', '')
                    ctx.output = f"💭 Recuerdo: {key} = {value}"
                    return ctx
                
                # Buscar en memoria semántica
                sem_results = agent.memory.semantic.search(key, k=1)
                if sem_results:
                    ctx.output = f"💭 {sem_results[0]['texto']}"
                    return ctx
                
                ctx.output = f"❓ No recuerdo '{key}'"
            except Exception as e:
                ctx.error = f"Error recordando: {e}"
    
    return ctx


def step_learning(ctx: Context, agent) -> Context:
    """Paso 3c: Aprender si es 'aprende X'"""
    if not ctx.allowed or ctx.parsed is None:
        return ctx
    
    if ctx.output is not None:
        return ctx
    
    if ctx.parsed.intent == 'learning':
        text = ctx.parsed.entities.get('text')
        if text:
            try:
                # Guardar como conocimiento
                key = f"learning_{len(agent.memory.long_term.get_all())}"
                agent.memory.add_knowledge(key, text, {'tipo': 'aprendizaje', 'original': text})
                ctx.learned = True
                ctx.output = f"🎓 Aprendido: {text}"
            except Exception as e:
                ctx.error = f"Error aprendiendo: {e}"
    
    return ctx


def step_tool_execution(ctx: Context, agent) -> Context:
    """Paso 4: Ejecutar herramienta (hora, fecha, math)"""
    if not ctx.allowed or ctx.parsed is None:
        return ctx
    
    if ctx.output is not None:
        return ctx
    
    intent = ctx.parsed.intent
    
    try:
        if intent == 'math':
            nums = ctx.parsed.entities.get('numbers', [])
            op = ctx.parsed.entities.get('operator', '+')
            if len(nums) >= 2:
                a, b = nums[0], nums[1]
                if op == '+': result = a + b
                elif op == '-': result = a - b
                elif op == '*': result = a * b
                elif op == '/': result = a / b if b != 0 else "Error"
                else: result = None
                
                if result is not None:
                    ctx.output = f"{a} {op} {b} = {result}"
        
        elif intent == 'time':
            from datetime import datetime
            hora = datetime.now().strftime('%H:%M:%S')
            ctx.output = f"🕐 {hora}"
        
        elif intent == 'date':
            from datetime import datetime
            fecha = datetime.now().strftime('%d/%m/%Y')
            ctx.output = f"📅 {fecha}"
        
        elif intent == 'greeting':
            ctx.output = "¡Hola! Soy Cerebro Zero 2.1. ¿En qué puedo ayudarte?"
        
        elif intent == 'farewell':
            ctx.output = "¡Hasta luego! Ha sido un placer."
    
    except Exception as e:
        ctx.error = f"Error en tool: {e}"
    
    return ctx


def step_memory_search(ctx: Context, agent) -> Context:
    """Paso 5: Buscar en memoria si no hay output"""
    if not ctx.allowed or ctx.output is not None:
        return ctx
    
    try:
        # Búsqueda exacta en episódica
        episodios = agent.memory.get_all_episodic()
        input_lower = ctx.input.lower().strip()
        for ep in episodios:
            ep_input = ep.get('input', '').lower().strip()
            if ep_input == input_lower:
                ctx.exact_match = ep.get('output', '')
                ctx.output = ctx.exact_match
                return ctx
        
        # Búsqueda semántica
        results = agent.memory.semantic.search(ctx.input, k=1)
        if results and results[0]['similitud'] > 0.5:
            ctx.output = results[0]['texto']
    except Exception:
        pass
    
    return ctx


def step_verify(ctx: Context, verifier) -> Context:
    """Paso 6: Verificar respuesta"""
    if ctx.output is None:
        ctx.output = "No tengo información sobre eso."
        ctx.confidence = 0.3
    else:
        try:
            evaluation = verifier.evaluar(ctx.input, ctx.output)
            ctx.confidence = evaluation.get('confianza', 0.5)
            ctx.verified = ctx.confidence >= 0.7
        except Exception:
            ctx.confidence = 0.5
    
    return ctx


def step_learn(ctx: Context, agent) -> Context:
    """Paso 7: Aprender de la experiencia"""
    if ctx.output and agent:
        try:
            correcto = ctx.verified or ctx.tool_result is not None or ctx.memory_stored
            agent.learning.aprender(ctx.input, ctx.output, correcto=correcto)
            ctx.learned = True
        except Exception:
            pass
    return ctx


if __name__ == "__main__":
    print("🧪 PIPELINE V3.0")
    print(f"   Pasos: parse, sandbox, memory_store, memory_recall, learning, tool, memory_search, verify, learn")
