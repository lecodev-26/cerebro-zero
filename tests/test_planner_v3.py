"""
Tests de Planner 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from reasoning.planner_v3 import (
    Planner3, Plan, Subgoal, Action, PLANTILLAS,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def planner():
    return Planner3()


@pytest.fixture
def plan_entrenar(planner):
    return planner.planificar("Entrenar el modelo y comprobar si mejoró")


# ============================================
# TESTS DE ACTION
# ============================================

def test_action_creation():
    a = Action("test", tool="dummy", args={"x": 1})
    assert a.nombre == "test"
    assert a.tool == "dummy"
    assert a.args == {"x": 1}


def test_action_to_dict():
    a = Action("test", tool="dummy", args={"x": 1}, descripcion="desc")
    d = a.to_dict()
    assert d["nombre"] == "test"
    assert d["tool"] == "dummy"


def test_action_from_dict():
    d = {"nombre": "test", "tool": "dummy", "args": {"x": 1}, "descripcion": ""}
    a = Action.from_dict(d)
    assert a.nombre == "test"


# ============================================
# TESTS DE SUBGOAL
# ============================================

def test_subgoal_creation():
    sg = Subgoal(id="a", descripcion="Hacer A")
    assert sg.id == "a"
    assert sg.completado == False
    assert sg.depende_de == []


def test_subgoal_to_dict():
    sg = Subgoal(id="a", descripcion="Hacer A", depende_de=["b"])
    d = sg.to_dict()
    assert d["id"] == "a"
    assert d["depende_de"] == ["b"]


def test_subgoal_from_dict():
    d = {"id": "a", "descripcion": "Hacer A", "accion": None,
         "depende_de": ["b"], "completado": True}
    sg = Subgoal.from_dict(d)
    assert sg.id == "a"
    assert sg.completado == True


# ============================================
# TESTS DE PLAN
# ============================================

def test_plan_creation():
    p = Plan(goal="test")
    assert p.goal == "test"
    assert p.subgoals == []
    assert p.completado == False


def test_plan_get_subgoal():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    sg = p.get_subgoal("a")
    assert sg is not None
    assert sg.id == "a"


def test_plan_get_subgoal_no_existe():
    p = Plan(goal="test")
    assert p.get_subgoal("no_existe") is None


def test_plan_pendientes_completados():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B"))
    p.subgoals[0].completado = True
    
    assert len(p.completados()) == 1
    assert len(p.pendientes()) == 1


def test_plan_progreso():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B"))
    assert p.progreso() == 0.0
    p.subgoals[0].completado = True
    assert p.progreso() == 0.5
    p.subgoals[1].completado = True
    assert p.progreso() == 1.0


def test_plan_progreso_vacio():
    p = Plan(goal="test")
    assert p.progreso() == 0.0


# ============================================
# TESTS DE ORDEN TOPOLÓGICO
# ============================================

def test_orden_topologico_simple():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B", depende_de=["a"]))
    p.subgoals.append(Subgoal(id="c", descripcion="C", depende_de=["b"]))
    
    orden = p.orden_topologico()
    assert orden.index("a") < orden.index("b")
    assert orden.index("b") < orden.index("c")


def test_orden_topologico_paralelo():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B"))
    p.subgoals.append(Subgoal(id="c", descripcion="C", depende_de=["a", "b"]))
    
    orden = p.orden_topologico()
    assert orden.index("a") < orden.index("c")
    assert orden.index("b") < orden.index("c")


def test_orden_topologico_ciclo_detectado():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A", depende_de=["b"]))
    p.subgoals.append(Subgoal(id="b", descripcion="B", depende_de=["a"]))
    
    with pytest.raises(ValueError, match="Ciclo"):
        p.orden_topologico()


def test_orden_topologico_dep_rota():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A", depende_de=["no_existe"]))
    
    with pytest.raises(ValueError, match="no existe"):
        p.orden_topologico()


# ============================================
# TESTS DE SIGUIENTE / PUEDE_EJECUTAR
# ============================================

def test_plan_siguiente():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B", depende_de=["a"]))
    
    sg = p.siguiente()
    assert sg.id == "a"


def test_plan_siguiente_respeta_deps():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B", depende_de=["a"]))
    
    # Completar A
    p.subgoals[0].completado = True
    
    # Siguiente debe ser B
    sg = p.siguiente()
    assert sg.id == "b"


def test_plan_siguiente_none_si_todo_hecho():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A", completado=True))
    assert p.siguiente() is None


def test_plan_puede_ejecutar():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.subgoals.append(Subgoal(id="b", descripcion="B", depende_de=["a"]))
    
    assert p.puede_ejecutar("a") == True
    assert p.puede_ejecutar("b") == False


def test_plan_puede_ejecutar_si_deps_ok():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A", completado=True))
    p.subgoals.append(Subgoal(id="b", descripcion="B", depende_de=["a"]))
    
    assert p.puede_ejecutar("b") == True


# ============================================
# TESTS DE MARCAR ESTADO
# ============================================

def test_plan_marcar_inicio():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.marcar_inicio("a")
    assert p.get_subgoal("a").timestamp_inicio is not None


def test_plan_marcar_completado():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.marcar_completado("a", resultado="ok")
    sg = p.get_subgoal("a")
    assert sg.completado == True
    assert sg.resultado == "ok"


def test_plan_marcar_completado_cierra_plan():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.marcar_completado("a")
    assert p.completado == True


def test_plan_marcar_error():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    p.marcar_error("a", "fallo")
    assert p.get_subgoal("a").error == "fallo"


# ============================================
# TESTS DE PLANNER3 - DETECCIÓN
# ============================================

def test_detectar_entrenar(planner):
    assert planner.detectar_intencion("Entrenar el modelo") == "entrenar"


def test_detectar_evaluar(planner):
    assert planner.detectar_intencion("evaluar el modelo") == "evaluar"


def test_detectar_recordar(planner):
    assert planner.detectar_intencion("recordar color") == "recordar"


def test_detectar_desconocido(planner):
    assert planner.detectar_intencion("hacer magia") is None


# ============================================
# TESTS DE PLANNER3 - PLANIFICAR
# ============================================

def test_planificar_entrenar(planner):
    plan = planner.planificar("Entrenar el modelo")
    assert len(plan.subgoals) == 8
    assert plan.goal == "Entrenar el modelo"


def test_planificar_evaluar(planner):
    plan = planner.planificar("evaluar el modelo")
    assert len(plan.subgoals) == 4


def test_planificar_recordar(planner):
    plan = planner.planificar("recordar color")
    assert len(plan.subgoals) == 3


def test_planificar_generico(planner):
    plan = planner.planificar("hacer magia")
    assert len(plan.subgoals) == 1
    assert plan.subgoals[0].id == "ejecutar"


def test_planificar_guarda_en_lista(planner):
    planner.planificar("Entrenar")
    planner.planificar("Evaluar")
    assert len(planner.planes) == 2


def test_planificar_valida_deps(planner):
    # Debería pasar sin problemas (plantillas válidas)
    plan = planner.planificar("Entrenar")
    assert plan is not None


# ============================================
# TESTS DE EJECUCIÓN
# ============================================

def test_ejecutar_siguiente_sin_tool(planner, plan_entrenar):
    r = planner.ejecutar_siguiente(plan_entrenar)
    assert r["ok"] == True
    assert r["subgoal_id"] == "cargar_dataset"


def test_ejecutar_plan_completo(planner, plan_entrenar):
    resultados = planner.ejecutar_plan(plan_entrenar)
    assert len(resultados) == 8
    assert plan_entrenar.completado == True
    assert plan_entrenar.progreso() == 1.0


def test_ejecutar_plan_orden_correcto(planner, plan_entrenar):
    resultados = planner.ejecutar_plan(plan_entrenar)
    ids = [r["subgoal_id"] for r in resultados]
    
    # cargar_dataset antes que entrenar
    assert ids.index("cargar_dataset") < ids.index("entrenar")
    # entrenar antes que evaluar_validation
    assert ids.index("entrenar") < ids.index("evaluar_validation")
    # generar_informe al final
    assert ids[-1] == "generar_informe"


def test_ejecutar_plan_ya_completado(planner, plan_entrenar):
    planner.ejecutar_plan(plan_entrenar)
    # Segunda ejecución no debería hacer nada
    resultados = planner.ejecutar_plan(plan_entrenar)
    assert len(resultados) == 0


# ============================================
# TESTS DE SERIALIZACIÓN
# ============================================

def test_plan_to_dict():
    p = Plan(goal="test")
    p.subgoals.append(Subgoal(id="a", descripcion="A"))
    d = p.to_dict()
    assert d["goal"] == "test"
    assert len(d["subgoals"]) == 1


def test_plan_from_dict():
    d = {
        "goal": "test",
        "subgoals": [{"id": "a", "descripcion": "A", "accion": None,
                      "depende_de": [], "completado": False}],
        "creado": 0,
        "completado": False,
    }
    p = Plan.from_dict(d)
    assert p.goal == "test"
    assert len(p.subgoals) == 1


def test_planner_guardar_cargar(planner, plan_entrenar):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        planner.guardar(plan_entrenar, tmp)
        plan2 = planner.cargar(tmp)
        assert plan2 is not None
        assert plan2.goal == plan_entrenar.goal
        assert len(plan2.subgoals) == len(plan_entrenar.subgoals)
    finally:
        os.unlink(tmp)


def test_planner_cargar_no_existe(planner):
    plan = planner.cargar("/ruta/que/no/existe.json")
    assert plan is None


# ============================================
# TESTS DE UTILIDADES
# ============================================

def test_planner_ultimo_plan(planner):
    planner.planificar("Entrenar")
    planner.planificar("Evaluar")
    ultimo = planner.ultimo_plan()
    assert ultimo.goal == "Evaluar"


def test_planner_ultimo_plan_vacio(planner):
    assert planner.ultimo_plan() is None


def test_planner_resumen(planner):
    planner.planificar("Entrenar")
    r = planner.resumen()
    assert r["total_planes"] == 1
    assert "entrenar" in r["plantillas"]


def test_planner_repr(planner):
    assert "Planner3" in repr(planner)


# ============================================
# TESTS DE PLANTILLAS
# ============================================

def test_plantillas_existen():
    assert "entrenar" in PLANTILLAS
    assert "evaluar" in PLANTILLAS
    assert "recordar" in PLANTILLAS


def test_plantilla_entrenar_valida():
    for item in PLANTILLAS["entrenar"]:
        assert "id" in item
        assert "desc" in item
        assert "deps" in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
