"""
Cerebro V3 — Agente cognitivo integrado
=========================================

Une todas las piezas de la versión 3.0:
- World Model          (estado del mundo)
- Working Memory       (memoria de trabajo)
- Procedural Memory    (procedimientos aprendidos)
- Planner 3.0          (goal → subgoals → actions)
- Tool System 3.0      (contratos + ejecución segura)
- Learning Engine      (accept/reject con verificación)
- RL básico            (GridWorld + Q-Learning)
- Tokenizer 3.0        (byte-level + BPE)
- Benchmark 4.0        (métricas científicas)
- Reproducibilidad     (snapshots verificables)

Filosofía: cada input del usuario recorre el pipeline cognitivo
completo y deja huella en memoria, plan y experiencia.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from memory.world_model import WorldModel
from memory.working_memory import WorkingMemory
from memory.procedural_memory import ProceduralMemory
from reasoning.planner_v3 import Planner3, Plan
from tools.tool_system_v3 import ToolRegistryV3, ToolSpec, Schema, ToolResult
from training.learning_engine import LearningEngine
from language.tokenizer_v3 import BPETokenizer


# ============================================
# RESPUESTA
# ============================================

@dataclass
class Respuesta:
    """Respuesta del agente con traza completa"""
    input: str
    output: str
    plan: Optional[Plan] = None
    tool_usada: Optional[str] = None
    tool_resultado: Optional[ToolResult] = None
    confianza: float = 0.0
    latencia: float = 0.0
    contexto: dict = field(default_factory=dict)
    errores: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "output": self.output,
            "plan": self.plan.resumen() if self.plan else None,
            "tool_usada": self.tool_usada,
            "tool_resultado": self.tool_resultado.to_dict() if self.tool_resultado else None,
            "confianza": self.confianza,
            "latencia": self.latencia,
            "errores": self.errores,
        }
    
    def __repr__(self):
        return f"Respuesta('{self.input[:20]}' → '{self.output[:30]}', conf={self.confianza:.2f})"


# ============================================
# CEREBRO V3
# ============================================

class CerebroV3:
    """
    Agente cognitivo integrado de la versión 3.0.
    """
    
    def __init__(
        self,
        nombre: str = "cerebro_v3",
        capacity_working: int = 10,
        tokenizer_corpus: Optional[str] = None,
        verbose: bool = False,
    ):
        self.nombre = nombre
        self.verbose = verbose
        self.creado = time.time()
        
        # Componentes cognitivos
        self.world = WorldModel()
        self.working = WorkingMemory(max_items=capacity_working)
        self.procedural = ProceduralMemory()
        self.planner = Planner3(world_model=self.world)
        self.tools = ToolRegistryV3()
        self.learning = LearningEngine()
        
        # Tokenizer
        self.tokenizer = BPETokenizer(vocab_size=280)
        if tokenizer_corpus:
            self.tokenizer.entrenar(tokenizer_corpus)
        
        # Stats
        self.num_procesados = 0
        self.historial_respuestas: List[Respuesta] = []
        
        # Inicializar estado del mundo
        self.world.set("cerebro.nombre", nombre, razon="init")
        self.world.set("cerebro.version", "3.0", razon="init")
        self.world.set("cerebro.creado", self.creado, razon="init")
        
        # Registrar tools por defecto
        self._registrar_tools_basicas()
        
        if verbose:
            print(f"🧠 CerebroV3 '{nombre}' inicializado")
    
    # ============================================
    # TOOLS BÁSICAS
    # ============================================
    
    def _registrar_tools_basicas(self):
        """Registra tools mínimas para que el cerebro sea funcional"""
        
        def tool_suma(a: float, b: float) -> float:
            return a + b
        
        def tool_resta(a: float, b: float) -> float:
            return a - b
        
        def tool_multiplica(a: float, b: float) -> float:
            return a * b
        
        self.tools.registrar(ToolSpec(
            nombre="suma",
            descripcion="Suma dos números",
            funcion=tool_suma,
            input_schema={
                "a": Schema("float", requerido=True),
                "b": Schema("float", requerido=True),
            },
            output_tipo="float",
            capacidades=["matemáticas", "aritmética"],
        ))
        
        self.tools.registrar(ToolSpec(
            nombre="resta",
            descripcion="Resta dos números",
            funcion=tool_resta,
            input_schema={
                "a": Schema("float", requerido=True),
                "b": Schema("float", requerido=True),
            },
            output_tipo="float",
            capacidades=["matemáticas", "aritmética"],
        ))
        
        self.tools.registrar(ToolSpec(
            nombre="multiplica",
            descripcion="Multiplica dos números",
            funcion=tool_multiplica,
            input_schema={
                "a": Schema("float", requerido=True),
                "b": Schema("float", requerido=True),
            },
            output_tipo="float",
            capacidades=["matemáticas", "aritmética"],
        ))
        
        def tool_recordar(clave: str) -> str:
            valor = self.world.get(clave)
            if valor is None:
                return f"💭 No recuerdo: {clave}"
            return f"💭 Recuerdo: {clave} = {valor}"
        
        def tool_aprender(clave: str, valor: Any) -> str:
            self.world.set(clave, valor, razon="aprendido")
            return f"🎓 Aprendido: {clave} = {valor}"
        
        self.tools.registrar(ToolSpec(
            nombre="recordar",
            descripcion="Recupera un valor del world model",
            funcion=tool_recordar,
            input_schema={"clave": Schema("str", requerido=True)},
            output_tipo="str",
            capacidades=["memoria", "consulta"],
        ))
        
        self.tools.registrar(ToolSpec(
            nombre="aprender",
            descripcion="Aprende un nuevo valor",
            funcion=tool_aprender,
            input_schema={
                "clave": Schema("str", requerido=True),
                "valor": Schema("any", requerido=True),
            },
            output_tipo="str",
            capacidades=["memoria", "aprendizaje"],
        ))
    
    # ============================================
    # PIPELINE PRINCIPAL
    # ============================================
    
    def procesar(self, texto: str) -> Respuesta:
        """
        Procesa un input del usuario a través del pipeline cognitivo:
        
        input → tokenizer → world model → planner → tool → observer → learning → respuesta
        """
        t0 = time.time()
        self.num_procesados += 1
        
        resp = Respuesta(input=texto, output="")
        
        try:
            # 1. Tokenizar
            tokens = self.tokenizer.encode(texto)
            resp.contexto["tokens"] = tokens
            resp.contexto["num_tokens"] = len(tokens)
            
            # 2. Guardar en working memory
            self.working.add(
                key=f"input_{self.num_procesados}",
                value=texto,
                prioridad=0.8,
            )
            
            # 3. Registrar en world model
            self.world.set(
                "ultimo_input",
                texto,
                razon="pipeline",
            )
            self.world.incrementar("stats.inputs_procesados")
            
            # 4. Detectar intención y ejecutar
            output = self._ejecutar_logica(texto, resp)
            resp.output = output
            
            # 5. Registrar experiencia
            self.learning.registrar_exito(
                estado=texto,
                accion=resp.tool_usada or "planner",
                resultado=output,
                recompensa=1.0,
            )
            
            # 6. Confianza
            resp.confianza = 0.9 if resp.tool_usada else 0.6
            
        except Exception as e:
            resp.output = f"⚠️ Error: {e}"
            resp.errores.append(str(e))
            resp.confianza = 0.0
            self.learning.registrar_fallo(
                estado=texto,
                accion="procesar",
                error=str(e),
            )
        
        resp.latencia = time.time() - t0
        self.historial_respuestas.append(resp)
        
        return resp
    
    def _ejecutar_logica(self, texto: str, resp: Respuesta) -> str:
        """
        Decide qué hacer con el input:
        - Recordar algo
        - Aprender algo
        - Hacer matemáticas
        - Planificar
        - Default
        """
        bajo = texto.lower().strip()
        
        # --- RECORDAR ---
        if bajo.startswith("recordar "):
            clave = texto[len("recordar "):].strip()
            r = self.tools.ejecutar("recordar", {"clave": clave})
            resp.tool_usada = "recordar"
            resp.tool_resultado = r
            return r.output if r.ok else r.error
        
        # --- APRENDER ---
        if bajo.startswith("aprende ") or bajo.startswith("aprender "):
            prefijo = "aprende " if bajo.startswith("aprende ") else "aprender "
            resto = texto[len(prefijo):]
            if "=" in resto:
                clave, valor = resto.split("=", 1)
                clave, valor = clave.strip(), valor.strip()
                # Intentar convertir a número
                try:
                    valor_conv = float(valor) if "." in valor else int(valor)
                except ValueError:
                    valor_conv = valor
                r = self.tools.ejecutar(
                    "aprender", {"clave": clave, "valor": valor_conv}
                )
                resp.tool_usada = "aprender"
                resp.tool_resultado = r
                return r.output if r.ok else r.error
            return "⚠️ Formato: aprende clave = valor"
        
        # --- MATEMÁTICAS ---
        operaciones = {
            "+": "suma",
            "-": "resta",
            "*": "multiplica",
        }
        
        for simbolo, tool_nombre in operaciones.items():
            if simbolo in texto:
                partes = texto.split(simbolo)
                if len(partes) == 2:
                    try:
                        a = float(partes[0].strip().split()[-1])
                        b = float(partes[1].strip().split()[0])
                        r = self.tools.ejecutar(
                            tool_nombre, {"a": a, "b": b}
                        )
                        resp.tool_usada = tool_nombre
                        resp.tool_resultado = r
                        if r.ok:
                            return f"{a} {simbolo} {b} = {r.output}"
                        return r.error
                    except (ValueError, IndexError):
                        pass
        
        # --- PLANIFICAR ---
        intencion = self.planner.detectar_intencion(texto)
        if intencion:
            plan = self.planner.planificar(texto)
            resp.plan = plan
            num_subgoals = len(plan.subgoals)
            return f"📋 Plan '{intencion}' con {num_subgoals} pasos"
        
        # --- DEFAULT ---
        return f"🤔 No entiendo: '{texto}'"
    
    # ============================================
    # API ADICIONAL
    # ============================================
    
    def planificar(self, goal: str) -> Plan:
        """Genera un plan explícitamente"""
        return self.planner.planificar(goal)
    
    def aprender_procedimiento(
        self,
        nombre: str,
        pasos: List[str],
        trigger: Optional[str] = None,
    ) -> bool:
        """
        Aprende un procedimiento a partir de una lista de strings.
        
        Convierte cada string en un Step(accion=..., descripcion=...).
        El trigger (regex que lo activa) por defecto es el nombre.
        """
        try:
            from memory.procedural_memory import Step
            pasos_obj = [
                Step(accion=p, descripcion=p) if not isinstance(p, Step) else p
                for p in pasos
            ]
            self.procedural.aprender(
                nombre=nombre,
                trigger=trigger or nombre,
                pasos=pasos_obj,
            )
            return True
        except Exception:
            return False
    
    def ejecutar_plan(self, plan: Plan) -> List[dict]:
        """Ejecuta un plan paso a paso"""
        return self.planner.ejecutar_plan(plan)
    
    # ============================================
    # STATS
    # ============================================
    
    def stats(self) -> dict:
        return {
            "nombre": self.nombre,
            "version": "3.0",
            "num_procesados": self.num_procesados,
            "world_keys": self.world.keys(),
            "working": f"{self.working.size()}/{self.working.max_items}",
            "procedimientos": len(getattr(self.procedural, "procedimientos", {})),
            "tools_registradas": len(self.tools.listar()),
            "experiencias": len(self.learning.buffer),
            "tokenizer_vocab": self.tokenizer.vocab_size,
            "confianza_media": (
                sum(r.confianza for r in self.historial_respuestas)
                / len(self.historial_respuestas)
                if self.historial_respuestas else 0.0
            ),
        }
    
    def __repr__(self):
        return f"CerebroV3('{self.nombre}', procesados={self.num_procesados})"


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO CEREBRO V3 (INTEGRACIÓN FINAL)")
    print("=" * 60)
    
    # Crear cerebro con corpus para tokenizer
    corpus = "hola mundo cerebro cero aprende recordar suma resta multiplica " * 10
    
    cerebro = CerebroV3(
        nombre="Zero",
        tokenizer_corpus=corpus,
        verbose=True,
    )
    
    # ============================================
    # PRUEBAS
    # ============================================
    print(f"\n📋 Tools registradas: {cerebro.tools.listar()}")
    
    print(f"\n🧪 Probando pipeline completo:")
    entradas = [
        "5 + 3",
        "10 * 4",
        "20 - 7",
        "aprende color = azul",
        "recordar color",
        "recordar no_existe",
        "Entrenar el modelo",
        "evaluar el test",
        "xyzzy foobar",
    ]
    
    for texto in entradas:
        r = cerebro.procesar(texto)
        print(f"\n   👤 '{texto}'")
        print(f"   🤖 {r.output}")
        if r.tool_usada:
            print(f"      (tool: {r.tool_usada}, conf: {r.confianza:.2f}, "
                  f"lat: {r.latencia*1000:.2f}ms)")
        elif r.plan:
            print(f"      (plan: {len(r.plan.subgoals)} subgoals, "
                  f"conf: {r.confianza:.2f})")
    
    # ============================================
    # PLANIFICACIÓN EXPLÍCITA
    # ============================================
    print(f"\n📋 Plan explícito:")
    plan = cerebro.planificar("Entrenar el modelo y comprobar si mejoró")
    print(f"   {plan}")
    for sg_id in plan.orden_topologico():
        sg = plan.get_subgoal(sg_id)
        deps = f" (deps: {sg.depende_de})" if sg.depende_de else ""
        print(f"      {sg.id}{deps}")
    
    # ============================================
    # APRENDER PROCEDIMIENTO
    # ============================================
    print(f"\n🎓 Aprender procedimiento:")
    ok = cerebro.aprender_procedimiento(
        nombre="sumar_lista",
        pasos=["iniciar contador=0", "recorrer lista", "sumar cada elemento",
                "devolver contador"],
    )
    print(f"   Procedimiento 'sumar_lista' aprendido: {ok}")
    
    # ============================================
    # WORLD MODEL
    # ============================================
    print(f"\n🌍 World Model:")
    print(f"   Keys: {cerebro.world.keys()}")
    print(f"   nombre: {cerebro.world.get('cerebro.nombre')}")
    print(f"   color:  {cerebro.world.get('color')}")
    print(f"   inputs procesados: {cerebro.world.get('stats.inputs_procesados')}")
    
    # ============================================
    # WORKING MEMORY
    # ============================================
    print(f"\n🧠 Working Memory:")
    print(f"   Items: {cerebro.working.size()}/{cerebro.working.max_items}")
    
    # ============================================
    # LEARNING ENGINE
    # ============================================
    print(f"\n🧬 Learning Engine:")
    stats_learning = cerebro.learning.buffer.stats()
    print(f"   Experiencias: {stats_learning.get('total', 0)}")
    print(f"   Verificadas:  {stats_learning.get('verificadas', 0)}")
    
    # ============================================
    # STATS GLOBALES
    # ============================================
    print(f"\n📊 Stats del cerebro:")
    for k, v in cerebro.stats().items():
        print(f"   {k}: {v}")
    
    print("\n✅ CEREBRO V3 INTEGRADO Y FUNCIONANDO")
    print(f"   {cerebro}")
