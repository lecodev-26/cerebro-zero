"""
Planner 3.0 — Planificación jerárquica con dependencias
========================================================

Filosofía:
    Goal → Subgoals → Dependencies → Actions → Expected results

El planner construye planes a partir de objetivos y puede:
- Descomponer en subgoals
- Detectar dependencias entre subgoals
- Ordenar topológicamente
- Ejecutar paso a paso
- Verificar progreso

Integración:
- Lee del World Model el estado actual
- Sabe usar tools del registry
- Persiste planes en JSON
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime


# ============================================
# ESTRUCTURAS DE DATOS
# ============================================

@dataclass
class Action:
    """Una acción ejecutable concreta"""
    nombre: str
    tool: Optional[str] = None          # tool del registry (si aplica)
    args: dict = field(default_factory=dict)
    descripcion: str = ""
    
    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "tool": self.tool,
            "args": self.args,
            "descripcion": self.descripcion,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            nombre=d["nombre"],
            tool=d.get("tool"),
            args=d.get("args", {}),
            descripcion=d.get("descripcion", ""),
        )


@dataclass
class Subgoal:
    """Un subobjetivo con dependencias y acción asociada"""
    id: str
    descripcion: str
    accion: Optional[Action] = None
    depende_de: List[str] = field(default_factory=list)
    completado: bool = False
    resultado: Any = None
    error: Optional[str] = None
    timestamp_inicio: Optional[float] = None
    timestamp_fin: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "descripcion": self.descripcion,
            "accion": self.accion.to_dict() if self.accion else None,
            "depende_de": self.depende_de,
            "completado": self.completado,
            "resultado": str(self.resultado)[:200] if self.resultado else None,
            "error": self.error,
            "timestamp_inicio": self.timestamp_inicio,
            "timestamp_fin": self.timestamp_fin,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Subgoal":
        return cls(
            id=d["id"],
            descripcion=d["descripcion"],
            accion=Action.from_dict(d["accion"]) if d.get("accion") else None,
            depende_de=d.get("depende_de", []),
            completado=d.get("completado", False),
            resultado=d.get("resultado"),
            error=d.get("error"),
            timestamp_inicio=d.get("timestamp_inicio"),
            timestamp_fin=d.get("timestamp_fin"),
        )


@dataclass
class Plan:
    """Conjunto ordenado de subgoals"""
    goal: str
    subgoals: List[Subgoal] = field(default_factory=list)
    creado: float = field(default_factory=time.time)
    completado: bool = False
    
    # ============================================
    # CONSULTAS
    # ============================================
    
    def get_subgoal(self, sid: str) -> Optional[Subgoal]:
        for sg in self.subgoals:
            if sg.id == sid:
                return sg
        return None
    
    def pendientes(self) -> List[Subgoal]:
        return [sg for sg in self.subgoals if not sg.completado]
    
    def completados(self) -> List[Subgoal]:
        return [sg for sg in self.subgoals if sg.completado]
    
    def progreso(self) -> float:
        if not self.subgoals:
            return 0.0
        return len(self.completados()) / len(self.subgoals)
    
    # ============================================
    # ORDEN TOPOLÓGICO
    # ============================================
    
    def orden_topologico(self) -> List[str]:
        """
        Devuelve los IDs de subgoals en orden topológico
        respetando dependencias (Kahn's algorithm).
        """
        # Construir grafo
        ids = [sg.id for sg in self.subgoals]
        deps = {sg.id: set(sg.depende_de) for sg in self.subgoals}
        
        # Verificar que no hay dependencias rotas
        for sid, d in deps.items():
            for dep in d:
                if dep not in ids:
                    raise ValueError(f"Subgoal {sid} depende de {dep} que no existe")
        
        # Kahn's algorithm
        orden = []
        pendientes = set(ids)
        
        while pendientes:
            sin_deps = [sid for sid in pendientes if not (deps[sid] & pendientes)]
            if not sin_deps:
                raise ValueError("Ciclo detectado en dependencias")
            # Ordenar alfabéticamente para determinismo
            sin_deps.sort()
            for sid in sin_deps:
                orden.append(sid)
                pendientes.remove(sid)
        
        return orden
    
    def siguiente(self) -> Optional[Subgoal]:
        """
        Devuelve el siguiente subgoal ejecutable
        (no completado y con dependencias satisfechas).
        """
        for sid in self.orden_topologico():
            sg = self.get_subgoal(sid)
            if sg.completado:
                continue
            # Comprobar que todas las dependencias están completadas
            deps_ok = all(
                self.get_subgoal(d).completado
                for d in sg.depende_de
            )
            if deps_ok:
                return sg
        return None
    
    def puede_ejecutar(self, sid: str) -> bool:
        """¿Se puede ejecutar este subgoal ahora?"""
        sg = self.get_subgoal(sid)
        if not sg or sg.completado:
            return False
        return all(
            self.get_subgoal(d).completado
            for d in sg.depende_de
        )
    
    # ============================================
    # MARCAR ESTADO
    # ============================================
    
    def marcar_inicio(self, sid: str):
        sg = self.get_subgoal(sid)
        if sg:
            sg.timestamp_inicio = time.time()
    
    def marcar_completado(self, sid: str, resultado: Any = None):
        sg = self.get_subgoal(sid)
        if sg:
            sg.completado = True
            sg.resultado = resultado
            sg.timestamp_fin = time.time()
            if all(s.completado for s in self.subgoals):
                self.completado = True
    
    def marcar_error(self, sid: str, error: str):
        sg = self.get_subgoal(sid)
        if sg:
            sg.error = error
            sg.timestamp_fin = time.time()
    
    # ============================================
    # SERIALIZACIÓN
    # ============================================
    
    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "subgoals": [sg.to_dict() for sg in self.subgoals],
            "creado": self.creado,
            "completado": self.completado,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        plan = cls(goal=d["goal"], creado=d.get("creado", time.time()))
        plan.subgoals = [Subgoal.from_dict(sg) for sg in d["subgoals"]]
        plan.completado = d.get("completado", False)
        return plan
    
    def resumen(self) -> dict:
        return {
            "goal": self.goal,
            "total_subgoals": len(self.subgoals),
            "completados": len(self.completados()),
            "pendientes": len(self.pendientes()),
            "progreso": f"{self.progreso()*100:.1f}%",
            "siguiente": self.siguiente().id if self.siguiente() else None,
        }
    
    def __repr__(self):
        return f"Plan('{self.goal}', {len(self.completados())}/{len(self.subgoals)})"


# ============================================
# PLANTILLAS DE PLANES (dominio cerrado)
# ============================================

PLANTILLAS = {
    "entrenar": [
        {
            "id": "cargar_dataset",
            "desc": "Cargar dataset de entrenamiento",
            "accion": Action("cargar_dataset", tool="dataset_loader"),
            "deps": [],
        },
        {
            "id": "medir_baseline",
            "desc": "Medir accuracy del modelo actual",
            "accion": Action("medir_baseline", tool="evaluator"),
            "deps": ["cargar_dataset"],
        },
        {
            "id": "entrenar",
            "desc": "Entrenar el modelo",
            "accion": Action("entrenar", tool="trainer"),
            "deps": ["cargar_dataset"],
        },
        {
            "id": "evaluar_validation",
            "desc": "Evaluar en conjunto de validación",
            "accion": Action("evaluar_validation", tool="evaluator"),
            "deps": ["entrenar"],
        },
        {
            "id": "comparar_baseline",
            "desc": "Comparar resultado con baseline",
            "accion": Action("comparar_baseline"),
            "deps": ["medir_baseline", "evaluar_validation"],
        },
        {
            "id": "ejecutar_test",
            "desc": "Ejecutar tests del modelo",
            "accion": Action("ejecutar_test", tool="tester"),
            "deps": ["entrenar"],
        },
        {
            "id": "guardar_checkpoint",
            "desc": "Guardar checkpoint del modelo",
            "accion": Action("guardar_checkpoint", tool="serializer"),
            "deps": ["evaluar_validation"],
        },
        {
            "id": "generar_informe",
            "desc": "Generar informe final",
            "accion": Action("generar_informe", tool="reporter"),
            "deps": ["comparar_baseline", "ejecutar_test", "guardar_checkpoint"],
        },
    ],
    "evaluar": [
        {
            "id": "cargar_modelo",
            "desc": "Cargar modelo entrenado",
            "accion": Action("cargar_modelo", tool="serializer"),
            "deps": [],
        },
        {
            "id": "cargar_test",
            "desc": "Cargar conjunto de test",
            "accion": Action("cargar_test", tool="dataset_loader"),
            "deps": [],
        },
        {
            "id": "evaluar",
            "desc": "Evaluar modelo en test",
            "accion": Action("evaluar", tool="evaluator"),
            "deps": ["cargar_modelo", "cargar_test"],
        },
        {
            "id": "reporte",
            "desc": "Generar reporte de evaluación",
            "accion": Action("reporte", tool="reporter"),
            "deps": ["evaluar"],
        },
    ],
    "recordar": [
        {
            "id": "buscar_memoria",
            "desc": "Buscar en memoria semántica",
            "accion": Action("buscar_memoria", tool="memory_search"),
            "deps": [],
        },
        {
            "id": "recuperar",
            "desc": "Recuperar los datos encontrados",
            "accion": Action("recuperar", tool="memory_retrieve"),
            "deps": ["buscar_memoria"],
        },
        {
            "id": "responder",
            "desc": "Formular respuesta con los datos",
            "accion": Action("responder"),
            "deps": ["recuperar"],
        },
    ],
}


# ============================================
# PLANNER 3.0
# ============================================

class Planner3:
    """
    Planner jerárquico con:
    - Detección de intención por keywords
    - Plantillas de planes
    - Orden topológico real
    - Ejecución paso a paso
    - Persistencia
    """
    
    def __init__(self, world_model=None, tool_registry=None):
        self.world_model = world_model
        self.tool_registry = tool_registry
        self.planes: List[Plan] = []
    
    # ============================================
    # DETECCIÓN DE INTENCIÓN
    # ============================================
    
    def detectar_intencion(self, goal: str) -> Optional[str]:
        """
        Detecta qué tipo de plan usar según keywords del goal.
        Devuelve el nombre de la plantilla o None.
        """
        g = goal.lower()
        
        # Entrenar / train
        if any(k in g for k in ["entrenar", "train", "aprender modelo"]):
            return "entrenar"
        
        # Evaluar / test
        if any(k in g for k in ["evaluar", "evaluate", "test", "probar modelo"]):
            return "evaluar"
        
        # Recordar / memoria
        if any(k in g for k in ["recordar", "recuperar", "buscar en memoria"]):
            return "recordar"
        
        return None
    
    # ============================================
    # CONSTRUCCIÓN DE PLANES
    # ============================================
    
    def planificar(self, goal: str) -> Plan:
        """
        Construye un Plan a partir de un objetivo.
        
        Uso:
            plan = planner.planificar("Entrenar el modelo y comprobar si mejoró")
        """
        tipo = self.detectar_intencion(goal)
        
        if tipo and tipo in PLANTILLAS:
            plan = self._construir_desde_plantilla(goal, PLANTILLAS[tipo])
        else:
            # Plan genérico: un solo subgoal
            plan = Plan(goal=goal)
            plan.subgoals.append(Subgoal(
                id="ejecutar",
                descripcion=f"Ejecutar: {goal}",
                accion=Action("ejecutar", args={"goal": goal}),
            ))
        
        # Validar orden topológico (detectar ciclos/deps rotas)
        plan.orden_topologico()
        
        self.planes.append(plan)
        return plan
    
    def _construir_desde_plantilla(self, goal: str, plantilla: list) -> Plan:
        plan = Plan(goal=goal)
        for item in plantilla:
            sg = Subgoal(
                id=item["id"],
                descripcion=item["desc"],
                accion=item.get("accion"),
                depende_de=list(item.get("deps", [])),
            )
            plan.subgoals.append(sg)
        return plan
    
    # ============================================
    # EJECUCIÓN
    # ============================================
    
    def ejecutar_siguiente(self, plan: Plan) -> dict:
        """
        Ejecuta el siguiente subgoal del plan.
        Devuelve {ok, subgoal_id, resultado, error}
        """
        sg = plan.siguiente()
        if not sg:
            return {"ok": False, "error": "No hay subgoals ejecutables"}
        
        plan.marcar_inicio(sg.id)
        
        # Sin tool registry → solo marcamos como completado
        if not self.tool_registry or not sg.accion or not sg.accion.tool:
            plan.marcar_completado(sg.id, resultado="simulado")
            return {
                "ok": True,
                "subgoal_id": sg.id,
                "resultado": "simulado",
            }
        
        # Con tool registry → intentar ejecutar
        try:
            resultado = self.tool_registry.execute_by_input(
                sg.accion.tool,
                sg.accion.args,
            )
            plan.marcar_completado(sg.id, resultado=resultado)
            return {
                "ok": True,
                "subgoal_id": sg.id,
                "resultado": resultado,
            }
        except Exception as e:
            plan.marcar_error(sg.id, str(e))
            return {
                "ok": False,
                "subgoal_id": sg.id,
                "error": str(e),
            }
    
    def ejecutar_plan(self, plan: Plan) -> List[dict]:
        """Ejecuta el plan completo (o hasta que falle)"""
        resultados = []
        while True:
            sg = plan.siguiente()
            if not sg:
                break
            r = self.ejecutar_siguiente(plan)
            resultados.append(r)
            if not r["ok"]:
                break
        return resultados
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    
    def guardar(self, plan: Plan, archivo: str):
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        with open(archivo, "w") as f:
            json.dump(plan.to_dict(), f, indent=2, default=str)
    
    def cargar(self, archivo: str) -> Optional[Plan]:
        if not os.path.exists(archivo):
            return None
        with open(archivo, "r") as f:
            data = json.load(f)
        plan = Plan.from_dict(data)
        self.planes.append(plan)
        return plan
    
    # ============================================
    # UTILIDADES
    # ============================================
    
    def ultimo_plan(self) -> Optional[Plan]:
        return self.planes[-1] if self.planes else None
    
    def resumen(self) -> dict:
        return {
            "total_planes": len(self.planes),
            "plantillas": list(PLANTILLAS.keys()),
            "world_model": self.world_model is not None,
            "tool_registry": self.tool_registry is not None,
        }
    
    def __repr__(self):
        return f"Planner3(planes={len(self.planes)})"


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO PLANNER 3.0")
    print("=" * 60)
    
    planner = Planner3()
    
    # Detección de intención
    print("\n🔍 Detección de intención:")
    for g in ["Entrenar el modelo", "evaluar test", "recordar color", "hacer algo raro"]:
        print(f"   '{g}' → {planner.detectar_intencion(g)}")
    
    # Plan de entrenamiento
    print("\n📋 Plan: entrenar el modelo")
    plan = planner.planificar("Entrenar el modelo y comprobar si mejoró")
    print(f"   {plan}")
    print(f"   Subgoals: {len(plan.subgoals)}")
    print(f"   Orden topológico:")
    for sid in plan.orden_topologico():
        sg = plan.get_subgoal(sid)
        deps = f" (deps: {sg.depende_de})" if sg.depende_de else ""
        print(f"      {sid}{deps}")
    
    # Siguiente
    print(f"\n▶️ Siguiente ejecutable:")
    sg = plan.siguiente()
    print(f"   {sg.id}: {sg.descripcion}")
    
    # Ejecutar todo
    print(f"\n🚀 Ejecutando plan completo:")
    resultados = planner.ejecutar_plan(plan)
    print(f"   Total ejecutados: {len(resultados)}")
    print(f"   Progreso: {plan.progreso()*100:.1f}%")
    print(f"   Completado: {plan.completado}")
    
    # Plan de evaluación
    print(f"\n📋 Plan: evaluar modelo")
    plan2 = planner.planificar("evaluar el modelo en test")
    print(f"   {plan2}")
    print(f"   Subgoals: {[sg.id for sg in plan2.subgoals]}")
    
    # Persistencia
    print(f"\n💾 Test persistencia:")
    tmp = "/data/data/com.termux/files/home/cerebro_zero/test_plan_tmp.json"
    planner.guardar(plan, tmp)
    plan3 = planner.cargar(tmp)
    print(f"   Cargado: {plan3}")
    print(f"   Mismo goal: {plan3.goal == plan.goal}")
    print(f"   Mismo progreso: {plan3.progreso() == plan.progreso()}")
    os.unlink(tmp)
    
    # Plan desconocido → plan genérico
    print(f"\n❓ Plan genérico (goal desconocido):")
    plan4 = planner.planificar("hacer algo raro")
    print(f"   {plan4}")
    print(f"   Subgoals: {[sg.id for sg in plan4.subgoals]}")
    
    print("\n✅ PLANNER 3.0 FUNCIONANDO")
    print(f"   Resumen: {planner.resumen()}")
