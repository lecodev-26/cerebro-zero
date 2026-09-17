"""
Tests de Learning Engine
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from training.learning_engine import (
    Experiencia, ExperienceBuffer, EvaluadorMejora,
    ResultadoEvaluacion, LearningEngine,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def buffer():
    return ExperienceBuffer(capacidad=10)


@pytest.fixture
def evaluador():
    return EvaluadorMejora(umbral=0.0)


@pytest.fixture
def engine():
    return LearningEngine(buffer_capacidad=50, umbral_mejora=0.0)


# ============================================
# TESTS DE EXPERIENCIA
# ============================================

def test_experiencia_creation():
    e = Experiencia(estado="s", accion="a", resultado="r")
    assert e.estado == "s"
    assert e.recompensa == 0.0
    assert e.verificado == False


def test_experiencia_to_dict():
    e = Experiencia(estado="s", accion="a", resultado="r", recompensa=1.0)
    d = e.to_dict()
    assert d["recompensa"] == 1.0
    assert "estado" in d


# ============================================
# TESTS DE BUFFER
# ============================================

def test_buffer_creation(buffer):
    assert len(buffer) == 0
    assert buffer.capacidad == 10


def test_buffer_add(buffer):
    e = Experiencia("s", "a", "r")
    buffer.add(e)
    assert len(buffer) == 1


def test_buffer_limite_capacidad(buffer):
    for i in range(20):
        buffer.add(Experiencia(f"s{i}", "a", "r"))
    assert len(buffer) == 10


def test_buffer_sample(buffer):
    for i in range(5):
        buffer.add(Experiencia(f"s{i}", "a", "r"))
    lote = buffer.sample(3)
    assert len(lote) == 3


def test_buffer_sample_vacio(buffer):
    assert buffer.sample(5) == []


def test_buffer_sample_mas_de_lo_que_hay(buffer):
    buffer.add(Experiencia("s", "a", "r"))
    lote = buffer.sample(10)
    assert len(lote) == 1


def test_buffer_sample_positivas(buffer):
    buffer.add(Experiencia("s1", "a", "r", recompensa=1.0))
    buffer.add(Experiencia("s2", "a", "r", recompensa=-1.0))
    buffer.add(Experiencia("s3", "a", "r", recompensa=0.5))
    
    positivas = buffer.sample_positivas(5)
    assert len(positivas) == 2


def test_buffer_sample_positivas_sin_positivas(buffer):
    buffer.add(Experiencia("s", "a", "r", recompensa=-1.0))
    assert buffer.sample_positivas(5) == []


def test_buffer_clear(buffer):
    buffer.add(Experiencia("s", "a", "r"))
    buffer.clear()
    assert len(buffer) == 0


def test_buffer_stats_vacio(buffer):
    stats = buffer.stats()
    assert stats["total"] == 0


def test_buffer_stats_con_datos(buffer):
    buffer.add(Experiencia("s1", "a", "r", recompensa=1.0, verificado=True))
    buffer.add(Experiencia("s2", "a", "r", recompensa=-0.5, error="fallo"))
    stats = buffer.stats()
    assert stats["total"] == 2
    assert stats["recompensa_media"] == 0.25
    assert stats["verificadas"] == 1
    assert stats["con_error"] == 1


def test_buffer_repr(buffer):
    assert "ExperienceBuffer" in repr(buffer)


# ============================================
# TESTS DE EVALUADOR
# ============================================

def test_evaluador_aceptar(evaluador):
    r = evaluador.evaluar(0.5, 0.8)
    assert r.aceptar == True
    assert r.mejora == pytest.approx(0.3)


def test_evaluador_rechazar(evaluador):
    r = evaluador.evaluar(0.8, 0.5)
    assert r.aceptar == False


def test_evaluador_rechazar_igual(evaluador):
    r = evaluador.evaluar(0.5, 0.5)
    assert r.aceptar == False


def test_evaluador_umbral():
    ev = EvaluadorMejora(umbral=0.1)
    # Mejora de 0.05 no supera umbral de 0.1
    r1 = ev.evaluar(0.5, 0.55)
    assert r1.aceptar == False
    # Mejora de 0.2 sí supera
    r2 = ev.evaluar(0.5, 0.7)
    assert r2.aceptar == True


def test_resultado_evaluacion_to_dict():
    r = ResultadoEvaluacion(True, 0.3, "mejora", 0.5, 0.8)
    d = r.to_dict()
    assert d["aceptar"] == True
    assert d["mejora"] == 0.3


# ============================================
# TESTS DE LEARNING ENGINE - CREACIÓN
# ============================================

def test_engine_creation(engine):
    assert engine.version_actual["version"] == 0
    assert engine.version_actual["score"] == 0.0


def test_engine_repr(engine):
    assert "LearningEngine" in repr(engine)


# ============================================
# TESTS DE REGISTRO
# ============================================

def test_engine_registrar_experiencia(engine):
    e = engine.registrar_experiencia("s", "a", "r", recompensa=1.0)
    assert len(engine.buffer) == 1
    assert e.recompensa == 1.0


def test_engine_registrar_exito(engine):
    e = engine.registrar_exito("s", "a", "r")
    assert e.verificado == True
    assert e.recompensa == 1.0


def test_engine_registrar_fallo(engine):
    e = engine.registrar_fallo("s", "a", "error!")
    assert e.error == "error!"
    assert e.recompensa == -0.5


# ============================================
# TESTS DE REPLAY
# ============================================

def test_engine_sample_replay(engine):
    for i in range(5):
        engine.registrar_experiencia(f"s{i}", "a", "r")
    lote = engine.sample_replay(3)
    assert len(lote) == 3


def test_engine_sample_exitos(engine):
    engine.registrar_exito("s1", "a", "r")
    engine.registrar_fallo("s2", "a", "err")
    exitos = engine.sample_exitos(5)
    assert len(exitos) == 1


# ============================================
# TESTS DE ACEPT/REJECT
# ============================================

def test_engine_aceptar_mejora(engine):
    r = engine.intentar_actualizar(0.5)
    assert r.aceptar == True
    assert engine.version_actual["version"] == 1
    assert engine.version_actual["score"] == 0.5


def test_engine_rechazar_empeora(engine):
    engine.intentar_actualizar(0.8)
    r = engine.intentar_actualizar(0.5)
    assert r.aceptar == False
    # Versión sigue siendo la 1 con score 0.8
    assert engine.version_actual["score"] == 0.8


def test_engine_rechazar_igual(engine):
    engine.intentar_actualizar(0.5)
    r = engine.intentar_actualizar(0.5)
    assert r.aceptar == False


def test_engine_historial_evaluaciones(engine):
    engine.intentar_actualizar(0.5)
    engine.intentar_actualizar(0.3)
    engine.intentar_actualizar(0.6)
    assert len(engine.historial_evaluaciones) == 3


def test_engine_versiones_aceptadas(engine):
    engine.intentar_actualizar(0.5)  # aceptada
    engine.intentar_actualizar(0.3)  # rechazada
    engine.intentar_actualizar(0.7)  # aceptada
    assert len(engine.versiones) == 2


# ============================================
# TESTS DE ENTRENAR CON REPLAY
# ============================================

def test_engine_entrenar_replay(engine):
    # Registrar experiencias
    for i in range(10):
        engine.registrar_exito(f"s{i}", "a", "r")
    
    contador = {"n": 0}
    def entrenar(lote):
        contador["n"] += 1
    
    def evaluar():
        return 0.9
    
    r = engine.entrenar_con_replay(entrenar, evaluar, n_replay=5)
    assert r["ok"] == True
    assert contador["n"] == 1


def test_engine_entrenar_replay_buffer_vacio(engine):
    def entrenar(lote):
        pass
    
    def evaluar():
        return 0.5
    
    r = engine.entrenar_con_replay(entrenar, evaluar)
    assert r["ok"] == False
    assert "vacío" in r["razon"].lower()


def test_engine_entrenar_replay_error(engine):
    engine.registrar_exito("s", "a", "r")
    
    def entrenar(lote):
        raise ValueError("error de entrenamiento")
    
    def evaluar():
        return 0.5
    
    r = engine.entrenar_con_replay(entrenar, evaluar)
    assert r["ok"] == False
    assert "Error" in r["razon"]


def test_engine_entrenar_replay_rechaza(engine):
    # Hacer que la primera actualización sea alta
    engine.intentar_actualizar(0.9)
    
    engine.registrar_exito("s", "a", "r")
    
    def entrenar(lote):
        pass
    
    def evaluar():
        return 0.5  # peor que 0.9
    
    r = engine.entrenar_con_replay(entrenar, evaluar)
    assert r["ok"] == False


# ============================================
# TESTS DE PERSISTENCIA
# ============================================

def test_engine_guardar(engine):
    engine.registrar_exito("s", "a", "r")
    engine.intentar_actualizar(0.5)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        engine.guardar(tmp)
        assert os.path.exists(tmp)
        # Comprobar que es JSON válido
        import json
        with open(tmp) as f:
            data = json.load(f)
        assert "version_actual" in data
        assert data["version_actual"]["version"] == 1
    finally:
        os.unlink(tmp)


# ============================================
# TESTS DE RESUMEN
# ============================================

def test_engine_resumen(engine):
    engine.registrar_exito("s", "a", "r")
    engine.intentar_actualizar(0.5)
    r = engine.resumen()
    assert "buffer" in r
    assert "version_actual" in r
    assert r["versiones_aceptadas"] == 1


def test_engine_resumen_sin_evaluaciones(engine):
    r = engine.resumen()
    assert r["ultima_evaluacion"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
