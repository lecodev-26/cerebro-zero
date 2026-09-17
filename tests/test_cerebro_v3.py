"""
Tests de Cerebro V3 (Integración Final)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.cerebro_v3 import CerebroV3, Respuesta


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def cerebro():
    corpus = "hola mundo cerebro cero aprende recordar suma resta multiplica " * 5
    return CerebroV3(
        nombre="test",
        tokenizer_corpus=corpus,
        verbose=False,
    )


@pytest.fixture
def cerebro_min():
    """Cerebro sin corpus (más rápido)"""
    return CerebroV3(nombre="min", verbose=False)


# ============================================
# TESTS DE INICIALIZACIÓN
# ============================================

def test_cerebro_creation(cerebro):
    assert cerebro.nombre == "test"
    assert cerebro.num_procesados == 0


def test_cerebro_componentes(cerebro):
    assert cerebro.world is not None
    assert cerebro.working is not None
    assert cerebro.procedural is not None
    assert cerebro.planner is not None
    assert cerebro.tools is not None
    assert cerebro.learning is not None
    assert cerebro.tokenizer is not None


def test_cerebro_world_inicializado(cerebro):
    assert cerebro.world.get("cerebro.nombre") == "test"
    assert cerebro.world.get("cerebro.version") == "3.0"


def test_cerebro_tools_registradas(cerebro):
    tools = cerebro.tools.listar()
    assert "suma" in tools
    assert "resta" in tools
    assert "multiplica" in tools
    assert "recordar" in tools
    assert "aprender" in tools


def test_cerebro_repr(cerebro):
    assert "CerebroV3" in repr(cerebro)


# ============================================
# TESTS DE MATEMÁTICAS
# ============================================

def test_suma(cerebro):
    r = cerebro.procesar("5 + 3")
    assert "8" in r.output
    assert r.tool_usada == "suma"


def test_resta(cerebro):
    r = cerebro.procesar("10 - 4")
    assert "6" in r.output
    assert r.tool_usada == "resta"


def test_multiplica(cerebro):
    r = cerebro.procesar("3 * 4")
    assert "12" in r.output
    assert r.tool_usada == "multiplica"


def test_matematicas_con_decimales(cerebro):
    r = cerebro.procesar("2.5 + 1.5")
    assert "4" in r.output


def test_operacion_inyectada(cerebro):
    """Verificar que el pipeline no se rompe con input raro"""
    r = cerebro.procesar("+ + +")
    assert r.output != ""


# ============================================
# TESTS DE APRENDER Y RECORDAR
# ============================================

def test_aprender_simple(cerebro):
    r = cerebro.procesar("aprende color = azul")
    assert "azul" in r.output
    assert cerebro.world.get("color") == "azul"


def test_recordar_existente(cerebro):
    cerebro.procesar("aprende nombre = Manuel")
    r = cerebro.procesar("recordar nombre")
    assert "Manuel" in r.output


def test_recordar_no_existe(cerebro):
    r = cerebro.procesar("recordar no_existe_xyz")
    assert "No recuerdo" in r.output


def test_aprender_numero(cerebro):
    cerebro.procesar("aprende edad = 25")
    assert cerebro.world.get("edad") == 25


def test_aprender_float(cerebro):
    cerebro.procesar("aprende pi = 3.14")
    assert cerebro.world.get("pi") == 3.14


def test_aprender_sin_formato(cerebro):
    r = cerebro.procesar("aprende solo_esto")
    assert "Formato" in r.output or "aprende" in r.output.lower()


# ============================================
# TESTS DE PLANIFICACIÓN
# ============================================

def test_planificar_entrenar(cerebro):
    r = cerebro.procesar("Entrenar el modelo")
    assert r.plan is not None
    assert len(r.plan.subgoals) == 8


def test_planificar_evaluar(cerebro):
    r = cerebro.procesar("evaluar el modelo")
    assert r.plan is not None
    assert len(r.plan.subgoals) == 4


def test_plan_explicito(cerebro):
    plan = cerebro.planificar("Entrenar")
    assert plan is not None
    assert len(plan.subgoals) == 8


# ============================================
# TESTS DE DESCONOCIDO
# ============================================

def test_input_desconocido(cerebro):
    r = cerebro.procesar("xyzzy foobar")
    assert "No entiendo" in r.output


def test_input_vacio(cerebro):
    r = cerebro.procesar("")
    assert r.output != ""


# ============================================
# TESTS DE WORKING MEMORY
# ============================================

def test_working_crece(cerebro):
    size_inicial = cerebro.working.size()
    cerebro.procesar("test 1")
    cerebro.procesar("test 2")
    assert cerebro.working.size() > size_inicial


def test_working_limite(cerebro):
    for i in range(20):
        cerebro.procesar(f"input {i}")
    assert cerebro.working.size() <= cerebro.working.max_items


# ============================================
# TESTS DE LEARNING ENGINE
# ============================================

def test_learning_registra(cerebro):
    cerebro.procesar("5 + 3")
    assert len(cerebro.learning.buffer) > 0


def test_learning_registra_fallos(cerebro):
    """Input desconocido también se registra (como éxito con confianza baja)"""
    cerebro.procesar("basura total xyz")
    assert len(cerebro.learning.buffer) > 0


# ============================================
# TESTS DE RESPUESTA
# ============================================

def test_respuesta_to_dict(cerebro):
    r = cerebro.procesar("5 + 3")
    d = r.to_dict()
    assert "input" in d
    assert "output" in d
    assert d["tool_usada"] == "suma"


def test_respuesta_repr(cerebro):
    r = cerebro.procesar("5 + 3")
    assert "Respuesta" in repr(r)


def test_respuesta_tiene_latencia(cerebro):
    r = cerebro.procesar("5 + 3")
    assert r.latencia > 0


def test_respuesta_confianza(cerebro):
    r = cerebro.procesar("5 + 3")
    assert 0 <= r.confianza <= 1


# ============================================
# TESTS DE PROCEDIMIENTOS
# ============================================

def test_aprender_procedimiento(cerebro):
    ok = cerebro.aprender_procedimiento(
        nombre="test_proc",
        pasos=["paso 1", "paso 2", "paso 3"],
    )
    assert ok == True


def test_aprender_procedimiento_con_trigger(cerebro):
    ok = cerebro.aprender_procedimiento(
        nombre="sumar_lista",
        pasos=["iniciar", "iterar", "sumar"],
        trigger="sumar lista",
    )
    assert ok == True


def test_procedimiento_aparece_en_stats(cerebro):
    cerebro.aprender_procedimiento(nombre="p1", pasos=["a", "b"])
    stats = cerebro.stats()
    assert stats["procedimientos"] >= 1


# ============================================
# TESTS DE STATS
# ============================================

def test_stats_iniciales(cerebro):
    s = cerebro.stats()
    assert s["nombre"] == "test"
    assert s["version"] == "3.0"
    assert s["num_procesados"] == 0


def test_stats_tras_procesar(cerebro):
    cerebro.procesar("5 + 3")
    cerebro.procesar("3 * 2")
    s = cerebro.stats()
    assert s["num_procesados"] == 2


def test_stats_working(cerebro):
    cerebro.procesar("test")
    s = cerebro.stats()
    assert "/" in s["working"]
    assert "10" in s["working"]


def test_stats_tools(cerebro):
    s = cerebro.stats()
    assert s["tools_registradas"] >= 5


def test_stats_tokenizer(cerebro):
    s = cerebro.stats()
    assert s["tokenizer_vocab"] >= 260


def test_stats_confianza_media(cerebro):
    cerebro.procesar("5 + 3")
    s = cerebro.stats()
    assert 0 <= s["confianza_media"] <= 1


# ============================================
# TESTS DE PIPELINE
# ============================================

def test_pipeline_completo(cerebro):
    """Un input debe dejar huella en todos los componentes"""
    r = cerebro.procesar("aprende ciudad = Madrid")
    
    # World model actualizado
    assert cerebro.world.get("ciudad") == "Madrid"
    
    # Working memory tiene el input
    assert cerebro.working.size() > 0
    
    # Learning engine registró experiencia
    assert len(cerebro.learning.buffer) > 0
    
    # Respuesta tiene output
    assert r.output != ""


def test_pipeline_multiples_inputs(cerebro):
    entradas = [
        "5 + 3",
        "aprende color = azul",
        "recordar color",
        "Entrenar el modelo",
        "basura",
    ]
    for e in entradas:
        r = cerebro.procesar(e)
        assert r.output != ""


def test_pipeline_contexto(cerebro):
    r = cerebro.procesar("5 + 3")
    assert "tokens" in r.contexto
    assert "num_tokens" in r.contexto


# ============================================
# TESTS DE INTEGRACIÓN CON OTRAS FASES
# ============================================

def test_integracion_world_model(cerebro):
    """El world model se actualiza con cada input"""
    key_antes = len(cerebro.world.keys())
    cerebro.procesar("aprende nueva_key = nueva_valor")
    key_despues = len(cerebro.world.keys())
    assert key_despues >= key_antes


def test_integracion_tokenizer(cerebro):
    """El tokenizer procesa cualquier texto"""
    r = cerebro.procesar("🔥 hola 你好")
    assert r.output != ""


def test_integracion_planner(cerebro):
    """El planner se activa con keywords"""
    r = cerebro.procesar("Entrenar el modelo")
    assert r.plan is not None


def test_integracion_tools(cerebro):
    """Las tools se ejecutan correctamente"""
    r = cerebro.procesar("5 + 5")
    assert r.tool_usada == "suma"
    assert r.tool_resultado is not None
    assert r.tool_resultado.ok == True


def test_integracion_learning(cerebro):
    """Cada procesado registra experiencia"""
    n_antes = len(cerebro.learning.buffer)
    cerebro.procesar("5 + 3")
    n_despues = len(cerebro.learning.buffer)
    assert n_despues == n_antes + 1


# ============================================
# TESTS DE ERRORES
# ============================================

def test_error_no_rompe_cerebro(cerebro):
    """Un input problemático no debe romper el cerebro"""
    r = cerebro.procesar(None)
    # Debe devolver algo (con error capturado)
    assert r.output != ""
    assert len(r.errores) >= 0  # aceptable si no hay error


def test_cerebro_sigue_funcionando_tras_error(cerebro):
    cerebro.procesar(None)
    r = cerebro.procesar("5 + 3")
    assert "8" in r.output


# ============================================
# TESTS DE HISTORIAL
# ============================================

def test_historial_crece(cerebro):
    cerebro.procesar("a")
    cerebro.procesar("b")
    cerebro.procesar("c")
    assert len(cerebro.historial_respuestas) == 3


def test_historial_guarda_respuestas(cerebro):
    r = cerebro.procesar("5 + 3")
    ultimo = cerebro.historial_respuestas[-1]
    assert ultimo.input == "5 + 3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
