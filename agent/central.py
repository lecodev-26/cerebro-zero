"""
Cerebro Central - Orquesta el pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.pipeline import (
    Context,
    step_parse, step_sandbox_check, step_tool_execution,
    step_memory_retrieval, step_reasoning, step_verify, step_learn,
)

from security.parser import Parser
from security.sandbox import Sandbox
from security.permissions import Permission, PERMISSIONS_STANDARD
from tools.registry import ToolRegistry
from memory.retrieval import MemoryRetrieval
from reasoning.planner import Razonamiento
from evaluation.self_evaluation import SelfEvaluation
from training.continuous_learning import ContinuousLearning


class CerebroCentral:
    """
    Cerebro Central: pipeline limpio con cada paso aislado.
    
    Uso:
        cerebro = CerebroCentral()
        output = cerebro.procesar("¿Cuánto es 5 + 3?")
    """
    
    def __init__(self, debug: bool = False):
        self.name = "Cerebro Central"
        self.version = "2.0.0"
        self.debug = debug
        
        # Componentes
        self.parser = Parser()
        self.sandbox = Sandbox(
            permissions=PERMISSIONS_STANDARD.copy(),
            strict=True,
        )
        self.registry = ToolRegistry(sandbox=self.sandbox)
        self.memory = MemoryRetrieval()
        self.reasoner = Razonamiento(verbose=debug)
        self.verifier = SelfEvaluation()
        self.learning = ContinuousLearning(archivo="cerebro_central.json")
        
        # Registrar herramientas básicas
        self._register_basic_tools()
        
        # Historial de ejecuciones
        self.history = []
    
    def _register_basic_tools(self):
        """Registra las herramientas básicas en el registry"""
        
        @self.registry.register(
            name="calculadora",
            description="Calcula operaciones matemáticas",
            permissions={Permission.EXECUTE_MATH},
        )
        def calculadora(a, b, op='+'):
            a, b = float(a), float(b)
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '/': return a / b if b != 0 else "Error: división por cero"
            return "Operación no válida"
        
        @self.registry.register(
            name="fecha",
            description="Devuelve la fecha actual",
            permissions={Permission.READ_SYSTEM_INFO},
        )
        def fecha():
            from datetime import datetime
            return datetime.now().strftime('%d/%m/%Y')
        
        @self.registry.register(
            name="hora",
            description="Devuelve la hora actual",
            permissions={Permission.READ_SYSTEM_INFO},
        )
        def hora():
            from datetime import datetime
            return datetime.now().strftime('%H:%M:%S')
    
    def procesar(self, input_text: str) -> str:
        """
        Procesa una entrada a través del pipeline completo.
        """
        # Crear contexto
        ctx = Context(input=input_text)
        
        # Ejecutar pipeline
        ctx = step_parse(ctx, self.parser)
        ctx = step_sandbox_check(ctx, self.sandbox)
        ctx = step_tool_execution(ctx, self.registry)
        ctx = step_memory_retrieval(ctx, self.memory)
        ctx = step_reasoning(ctx, self.reasoner)
        ctx = step_verify(ctx, self.verifier)
        ctx = step_learn(ctx, self.learning)
        
        # Guardar en historial
        self.history.append(ctx)
        
        # Debug
        if self.debug:
            self._print_debug(ctx)
        
        # Devolver respuesta
        if not ctx.allowed:
            return f"🔒 Bloqueado por seguridad: {ctx.block_reason}"
        
        if ctx.output:
            return ctx.output
        
        return "No tengo información sobre eso."
    
    def _print_debug(self, ctx: Context):
        """Imprime información de debug del pipeline"""
        print(f"\n🔍 DEBUG del pipeline:")
        print(f"   Input: {ctx.input}")
        print(f"   Intent: {ctx.parsed.intent if ctx.parsed else 'N/A'}")
        print(f"   Allowed: {ctx.allowed}")
        print(f"   Tool result: {ctx.tool_result}")
        print(f"   Memory results: {len(ctx.memory_results)}")
        print(f"   Reasoning steps: {len(ctx.reasoning_steps)}")
        print(f"   Output: {ctx.output}")
        print(f"   Confidence: {ctx.confidence}")
        print(f"   Learned: {ctx.learned}")
    
    def enseñar(self, entrada: str, respuesta: str):
        """Enseña una respuesta al cerebro"""
        self.memory.add_experience(entrada, respuesta)
        self.learning.aprender(entrada, respuesta, correcto=True)
        print(f"🎓 Enseñado: {entrada} → {respuesta}")
    
    def resumen(self):
        """Resumen del estado del cerebro"""
        print(f"\n🧠 {self.name} v{self.version}")
        print(f"   Herramientas: {len(self.registry.tools)}")
        print(f"   Permisos: {len(self.sandbox.permissions)}")
        print(f"   Historial: {len(self.history)}")
        print(f"   Experiencias: {len(self.learning.experiencias)}")
    
    def __repr__(self):
        return f"CerebroCentral(v{self.version})"


if __name__ == "__main__":
    print("🧪 PROBANDO CEREBRO CENTRAL")
    print("="*50)
    
    cerebro = CerebroCentral(debug=False)
    
    # Enseñar
    cerebro.enseñar("hola", "Hola, soy Cerebro Central 2.0")
    cerebro.enseñar("adiós", "Hasta luego, ha sido un placer")
    
    # Procesar
    entradas = [
        "hola",
        "¿Cuánto es 5 + 3?",
        "¿Qué hora es?",
        "¿Qué fecha es hoy?",
        "adiós",
        "algo que no sé",
        "rm -rf /",
    ]
    
    for entrada in entradas:
        print(f"\n📝 Entrada: {entrada}")
        resultado = cerebro.procesar(entrada)
        print(f"   Respuesta: {resultado}")
    
    # Resumen
    cerebro.resumen()
    
    print("\n✅ CEREBRO CENTRAL FUNCIONANDO")
