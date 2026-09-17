"""
Tests del Planner + Verifier
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from reasoning.planner import Planner, Verifier, Plan, Step, Razonamiento


# ============================================
# TESTS DEL PLANNER
# ============================================

def test_plan_math():
    p = Planner()
    plan = p.planificar("¿Cuánto es 5 + 3?")
    assert len(plan.steps) == 1
    assert plan.steps[0].kind == 'math'


def test_plan_math_multiplication():
    p = Planner()
    plan = p.planificar("10 * 7")
    assert plan.steps[0].kind == 'math'


def test_plan_definition():
    p = Planner()
    plan = p.planificar("¿Qué es la IA?")
    assert plan.steps[0].kind == 'definition'


def test_plan_time():
    p = Planner()
    plan = p.planificar("¿Qué hora es?")
    assert plan.steps[0].kind == 'tool'


def test_plan_date():
    p = Planner()
    plan = p.planificar("¿Qué fecha es hoy?")
    assert plan.steps[0].kind == 'tool'


def test_plan_unknown():
    p = Planner()
    plan = p.planificar("algo random sin sentido")
    assert plan.steps[0].kind == 'unknown'


def test_plan_run_math():
    p = Planner()
    plan = p.planificar("5 + 3")
    result = plan.run()
    assert result == 8


def test_plan_run_math_mul():
    p = Planner()
    plan = p.planificar("10 * 7")
    result = plan.run()
    assert result == 70


def test_plan_run_math_div():
    p = Planner()
    plan = p.planificar("15 / 3")
    result = plan.run()
    assert result == 5.0


def test_plan_definition_with_knowledge():
    p = Planner()
    p.set_knowledge({'ia': 'Inteligencia Artificial'})
    plan = p.planificar("¿Qué es la IA?")
    result = plan.run()
    assert result == "Inteligencia Artificial"


def test_plan_definition_without_knowledge():
    p = Planner()
    plan = p.planificar("¿Qué es algo_que_no_está?")
    result = plan.run()
    assert result is None


# ============================================
# TESTS DEL VERIFIER
# ============================================

def test_verifier_math_ok():
    p = Planner()
    v = Verifier()
    plan = p.planificar("5 + 3")
    plan.run()
    ok, conf, _ = v.verificar(plan.final_answer, plan)
    assert ok
    assert conf > 0.9


def test_verifier_empty_response():
    p = Planner()
    v = Verifier()
    plan = p.planificar("5 + 3")
    ok, conf, _ = v.verificar(None, plan)
    assert not ok
    assert conf == 0.0


def test_verifier_negative_response():
    p = Planner()
    v = Verifier()
    plan = p.planificar("¿Qué es X?")
    ok, conf, _ = v.verificar("No sé qué es eso", plan)
    assert not ok
    assert conf < 0.5


def test_verifier_tool():
    p = Planner()
    v = Verifier()
    plan = p.planificar("¿Qué hora es?")
    plan.run()
    ok, conf, _ = v.verificar(plan.final_answer, plan)
    assert ok


def test_verifier_unknown_step():
    p = Planner()
    v = Verifier()
    plan = p.planificar("algo random")
    ok, conf, _ = v.verificar("hola", plan)
    assert not ok


# ============================================
# TESTS DE RAZONAMIENTO (interfaz unificada)
# ============================================

def test_razonamiento_math():
    r = Razonamiento(verbose=False)
    result = r.razonar("¿Cuánto es 5 + 3?")
    assert len(result) == 1
    assert "8" in result[0]


def test_razonamiento_definition():
    r = Razonamiento(verbose=False)
    r.set_knowledge({'ia': 'Inteligencia Artificial'})
    result = r.razonar("¿Qué es la IA?")
    assert len(result) == 1
    assert "Inteligencia" in result[0]


def test_razonamiento_unknown():
    r = Razonamiento(verbose=False)
    result = r.razonar("algo sin sentido")
    assert result == []


def test_razonamiento_time():
    r = Razonamiento(verbose=False)
    result = r.razonar("¿Qué hora es?")
    assert len(result) == 1
    assert ":" in result[0]


def test_razonamiento_history():
    r = Razonamiento(verbose=False)
    r.razonar("5 + 3")
    r.razonar("10 * 7")
    assert len(r.historial) == 2


def test_razonamiento_no_fake_response():
    """El razonador NO debe devolver 'He recibido tu pregunta'"""
    r = Razonamiento(verbose=False)
    result = r.razonar("algo random")
    # Antes devolvía "He recibido..."; ahora debe devolver []
    assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
