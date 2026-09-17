"""
Pipeline del Cerebro Central
Cada paso es una función pura que recibe un contexto y lo transforma.
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
    exact_match: Optional[str] = None  # Coincidencia exacta en memoria
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
            'dangerous': self.parsed.dangerous if self.parsed else None,
            'allowed': self.allowed,
            'block_reason': self.block_reason,
            'tool_result': str(self.tool_result) if self.tool_result else None,
            'memory_count': len(self.memory_results),
            'exact_match': self.exact_match,
            'reasoning_count': len(self.reasoning_steps),
            'verified': self.verified,
            'confidence': self.confidence,
            'output': self.output,
            'error': self.error,
            'learned': self.learned,
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
    """Paso 2: Verificar seguridad con sandbox"""
    if ctx.parsed is None:
        ctx.error = "No hay parsed para verificar"
        return ctx
    
    if ctx.parsed.dangerous:
        ctx.allowed = False
        ctx.block_reason = ctx.parsed.danger_reason
        return ctx
    
    ctx.allowed = True
    return ctx


def step_tool_execution(ctx: Context, tool_registry) -> Context:
    """Paso 3: Ejecutar herramienta si es necesario"""
    if not ctx.allowed or ctx.parsed is None:
        return ctx
    
    tool_name = _detect_tool_from_intent(ctx.parsed)
    
    if tool_name and tool_registry:
        try:
            args = _extract_args_for_tool(ctx.parsed, tool_name)
            if args:
                result = tool_registry.execute(tool_name, *args)
                if result.get('success'):
                    ctx.tool_result = result.get('result')
        except Exception as e:
            ctx.error = f"Error ejecutando {tool_name}: {e}"
    
    return ctx


def step_memory_retrieval(ctx: Context, memory) -> Context:
    """
    Paso 4: Recuperar de memoria.
    - Primero busca coincidencia exacta en episódica.
    - Luego busca por similitud.
    """
    # 1. Coincidencia exacta en memoria episódica
    try:
        episodios = memory.get_all_episodic()
        input_lower = ctx.input.lower().strip()
        for ep in episodios:
            ep_input = ep.get('input', '').lower().strip()
            if ep_input == input_lower:
                ctx.exact_match = ep.get('output', '')
                break
    except Exception:
        pass
    
    # 2. Si no hay exacta, buscar por similitud
    if not ctx.exact_match:
        try:
            results = memory.remember(ctx.input)
            if results:
                for key, items in results.items():
                    if isinstance(items, list):
                        ctx.memory_results.extend(items)
        except Exception:
            pass
    
    return ctx


def step_reasoning(ctx: Context, reasoner) -> Context:
    """
    Paso 5: Razonamiento paso a paso.
    SOLO si no hay tool_result, exact_match, ni memoria relevante.
    """
    # Si ya hay resultado, no razonar
    if ctx.tool_result is not None:
        return ctx
    
    if ctx.exact_match:
        return ctx
    
    if ctx.memory_results:
        for mem in ctx.memory_results:
            if isinstance(mem, dict) and mem.get('output'):
                return ctx
            if isinstance(mem, str) and len(mem) > 0:
                return ctx
    
    # Razonar
    try:
        steps = reasoner.razonar(ctx.input)
        if steps:
            ctx.reasoning_steps = steps
    except Exception:
        pass
    return ctx


def step_verify(ctx: Context, verifier) -> Context:
    """Paso 6: Verificar la respuesta"""
    if ctx.output is None:
        ctx.output = _pick_best_response(ctx)
    
    if ctx.output:
        try:
            evaluation = verifier.evaluar(ctx.input, ctx.output)
            ctx.confidence = evaluation.get('confianza', 0.0)
            ctx.verified = ctx.confidence >= 0.7
        except Exception:
            pass
    
    return ctx


def step_learn(ctx: Context, learning) -> Context:
    """Paso 7: Guardar experiencia"""
    if ctx.output and learning:
        try:
            correcto = ctx.verified or ctx.tool_result is not None or ctx.exact_match is not None
            learning.aprender(ctx.input, ctx.output, correcto=correcto)
            ctx.learned = True
        except Exception:
            pass
    return ctx


# ============================================
# HELPERS
# ============================================

def _detect_tool_from_intent(parsed) -> Optional[str]:
    """Detecta qué herramienta usar según la intención"""
    intent = parsed.intent
    if intent == 'math':
        if 'numbers' in parsed.entities and len(parsed.entities['numbers']) >= 2:
            return 'calculadora'
    return None


def _extract_args_for_tool(parsed, tool_name: str) -> list:
    """Extrae argumentos para una herramienta según el parsed"""
    if tool_name == 'calculadora' and 'numbers' in parsed.entities:
        nums = parsed.entities['numbers']
        op = parsed.entities.get('operator', '+')
        if len(nums) >= 2:
            return [nums[0], nums[1], op]
    return []


def _pick_best_response(ctx: Context) -> Optional[str]:
    """Elige la mejor respuesta disponible (por prioridad)"""
    # 1. Resultado de herramienta
    if ctx.tool_result is not None:
        return str(ctx.tool_result)
    
    # 2. Coincidencia EXACTA en memoria
    if ctx.exact_match:
        return ctx.exact_match
    
    # 3. Memoria por similitud
    if ctx.memory_results:
        for mem in ctx.memory_results:
            if isinstance(mem, dict) and mem.get('output'):
                return mem['output']
            if isinstance(mem, str) and len(mem) > 0:
                return mem
    
    # 4. Razonamiento
    if ctx.reasoning_steps:
        for step in ctx.reasoning_steps:
            if isinstance(step, str):
                return step
            if step is not None:
                return str(step)
    
    return None


if __name__ == "__main__":
    print("🧪 PIPELINE — Módulo cargado correctamente")
    print(f"   Context: {Context}")
    print(f"   Pasos: parse, sandbox, tool, memory, reasoning, verify, learn")
