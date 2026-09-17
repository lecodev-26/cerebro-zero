"""
Cerebro Central V3.0 con memoria y aprendizaje
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.pipeline import (
    Context,
    step_parse, step_sandbox_check,
    step_memory_store, step_memory_recall, step_learning,
    step_tool_execution, step_memory_search,
    step_verify, step_learn,
)

from security.parser import Parser
from security.sandbox import Sandbox
from security.permissions import Permission, PERMISSIONS_STANDARD
from memory.retrieval import MemoryRetrieval
from evaluation.objective_evaluator import ObjectiveEvaluator
from training.continuous_learning import ContinuousLearning


class CerebroCentral:
    """Cerebro Central V3.0 con memoria y aprendizaje"""
    
    def __init__(self, debug: bool = False):
        self.name = "Cerebro Central"
        self.version = "3.0.0"
        self.debug = debug
        
        self.parser = Parser()
        self.sandbox = Sandbox(
            permissions=PERMISSIONS_STANDARD.copy(),
            strict=True,
        )
        self.memory = MemoryRetrieval()
        self.verifier = ObjectiveEvaluator()
        self.learning = ContinuousLearning(archivo="cerebro_central_v3.json")
        
        self.history = []
    
    def procesar(self, input_text: str) -> str:
        """Procesa una entrada a través del pipeline"""
        ctx = Context(input=input_text)
        
        # Pipeline
        ctx = step_parse(ctx, self.parser)
        ctx = step_sandbox_check(ctx, self.sandbox)
        
        # Solo si no está bloqueado
        if ctx.allowed:
            ctx = step_memory_store(ctx, self)      # Guardar
            ctx = step_memory_recall(ctx, self)     # Recuperar
            ctx = step_learning(ctx, self)          # Aprender
            ctx = step_tool_execution(ctx, self)    # Herramientas
            ctx = step_memory_search(ctx, self)     # Búsqueda
        
        ctx = step_verify(ctx, self.verifier)
        ctx = step_learn(ctx, self)
        
        self.history.append(ctx)
        
        if self.debug:
            self._print_debug(ctx)
        
        if not ctx.allowed:
            return f"🔒 Bloqueado por seguridad: {ctx.block_reason}"
        
        return ctx.output or "No tengo información sobre eso."
    
    def _print_debug(self, ctx: Context):
        print(f"\n🔍 DEBUG:")
        print(f"   Input: {ctx.input}")
        print(f"   Intent: {ctx.parsed.intent if ctx.parsed else 'N/A'}")
        print(f"   Entities: {ctx.parsed.entities if ctx.parsed else '{}'}")
        print(f"   Allowed: {ctx.allowed}")
        print(f"   Output: {ctx.output}")
        print(f"   Confidence: {ctx.confidence}")
    
    def enseñar(self, entrada: str, respuesta: str):
        """Enseña una respuesta directa"""
        self.memory.add_experience(entrada, respuesta)
        self.learning.aprender(entrada, respuesta, correcto=True)
    
    def resumen(self):
        print(f"\n🧠 {self.name} v{self.version}")
        print(f"   Historial: {len(self.history)}")
        print(f"   Experiencias: {len(self.learning.experiencias)}")
        print(f"   Memoria: {self.memory}")
    
    def __repr__(self):
        return f"CerebroCentral(v{self.version})"


if __name__ == "__main__":
    print("🧪 PROBANDO CEREBRO CENTRAL V3.0")
    print("="*60)
    
    cerebro = CerebroCentral()
    
    # Enseñar
    cerebro.enseñar("hola", "¡Hola! Soy Cerebro Zero 2.1")
    
    # Probar
    entradas = [
        "hola",
        "¿Cuánto es 5 + 3?",
        "¿Cuánto es 10 * 7?",
        "recuerda color = azul",
        "recordar color",
        "recuerda nombre = Manuel",
        "recordar nombre",
        "aprende a programar en python",
        "¿Qué hora es?",
        "dime la hora",
        "¿Qué fecha es hoy?",
        "qué es la ia",
        "adiós",
        "rm -rf /",
    ]
    
    for entrada in entradas:
        print(f"\n📝 Entrada: {entrada}")
        resultado = cerebro.procesar(entrada)
        print(f"   → {resultado}")
    
    print("\n")
    cerebro.resumen()
    
    print("\n✅ CEREBRO CENTRAL V3.0 FUNCIONANDO")
