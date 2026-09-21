"""
Tests de Learning Transactions (FASE 4.3)
==========================================

Verifica que el Learning Engine implementa:
- snapshot() — captura de pesos
- rollback() — restauración real
- entrenar_con_replay_con_rollback() — commit/rollback transaccional

Regla de oro (ahora con verificación de pesos):
    El modelo NUNCA queda modificado si el score empeora.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from training.learning_engine import LearningEngine


# ============================================
# MOCKS
# ============================================

class TensorMock:
    """Mock mínimo de Tensor con .data y .copy()"""
    def __init__(self, v):
        self.data = v
    def copy(self):
        return TensorMock(self.data)


class MockModel:
    """Mock con .parameters() que devuelve tensores"""
    def __init__(self, pesos=None):
        self._pesos = [TensorMock(v) for v in (pesos or [1.0, 2.0])]
    def parameters(self):
        return self._pesos


class MockModelSinParameters:
    """Mock sin .parameters(), pero con __dict__"""
    def __init__(self):
        self.w = TensorMock(1.0)
        self.b = TensorMock(2.0)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def engine():
    e = LearningEngine(umbral_mejora=0.0)
    e.registrar_exito("estado", "acción", "resultado")
    return e


@pytest.fixture
def modelo():
    return MockModel([1.0, 2.0, 3.0])


# ============================================
# TESTS DE SNAPSHOT
# ============================================

def test_snapshot_creation(engine, modelo):
    snap = engine.snapshot(modelo)
    assert snap is not None
    assert snap["tipo"] == "parameters"
    assert len(snap["datos"]) == 3


def test_snapshot_copia_profunda(engine, modelo):
    """Modificar el modelo NO debe afectar el snapshot"""
    snap = engine.snapshot(modelo)
    modelo.parameters()[0].data = 999.0
    assert snap["datos"][0] == 1.0


def test_snapshot_modelo_sin_parameters(engine):
    """Fallback cuando no hay .parameters()"""
    m = MockModelSinParameters()
    snap = engine.snapshot(m)
    assert snap is not None
    assert "tipo" in snap


# ============================================
# TESTS DE ROLLBACK
# ============================================

def test_rollback_restaura_pesos(engine, modelo):
    snap = engine.snapshot(modelo)
    modelo.parameters()[0].data = 999.0
    modelo.parameters()[1].data = 888.0
    
    ok = engine.rollback(modelo, snap)
    assert ok == True
    assert modelo.parameters()[0].data == 1.0
    assert modelo.parameters()[1].data == 2.0


def test_rollback_sin_snapshot(engine, modelo):
    ok = engine.rollback(modelo, None)
    assert ok == False


def test_rollback_varios_pesos(engine):
    m = MockModel([10.0, 20.0, 30.0, 40.0])
    snap = engine.snapshot(m)
    
    for p in m.parameters():
        p.data = 0.0
    
    engine.rollback(m, snap)
    assert [p.data for p in m.parameters()] == [10.0, 20.0, 30.0, 40.0]


# ============================================
# TESTS DE COMMIT
# ============================================

def test_commit_cuando_mejora(engine, modelo):
    scores = iter([0.5, 0.9])
    
    def evaluar():
        return next(scores)
    
    def entrenar(lote):
        modelo.parameters()[0].data = 42.0
    
    r = engine.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert r["ok"] == True
    assert "COMMIT" in r["razon"]
    assert r["rollback"] == False
    # El modelo DEBE tener el nuevo peso
    assert modelo.parameters()[0].data == 42.0


def test_commit_actualiza_version(engine, modelo):
    engine.version_actual["score"] = 0.5
    scores = iter([0.5, 0.9])
    
    def evaluar():
        return next(scores)
    
    def entrenar(lote):
        pass
    
    r = engine.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert engine.version_actual["version"] == 1
    assert engine.version_actual["score"] == 0.9


# ============================================
# TESTS DE ROLLBACK TRANSACCIONAL
# ============================================

def test_rollback_cuando_empeora(engine, modelo):
    """El caso crítico: score empeora → el modelo DEBE volver al estado original"""
    pesos_originales = [p.data for p in modelo.parameters()]
    
    scores = iter([0.9, 0.1])  # empeora
    
    def evaluar():
        return next(scores)
    
    def entrenar(lote):
        # El modelo se modifica DURANTE el entrenamiento
        for p in modelo.parameters():
            p.data = 999.0
    
    r = engine.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert r["ok"] == False
    assert "ROLLBACK" in r["razon"]
    assert r["rollback"] == True
    # El modelo DEBE estar en su estado original
    pesos_finales = [p.data for p in modelo.parameters()]
    assert pesos_finales == pesos_originales


def test_rollback_no_cambia_version(engine, modelo):
    engine.version_actual["version"] = 5
    engine.version_actual["score"] = 0.9
    
    scores = iter([0.9, 0.1])
    
    def evaluar():
        return next(scores)
    
    def entrenar(lote):
        pass
    
    r = engine.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert engine.version_actual["version"] == 5  # sin cambios
    assert engine.version_actual["score"] == 0.9


def test_rollback_con_muchos_pesos(engine):
    """Verifica que rollback funciona con modelos grandes"""
    m = MockModel(list(range(100)))
    pesos_originales = [p.data for p in m.parameters()]
    
    scores = iter([0.5, 0.1])
    def evaluar(): return next(scores)
    
    def entrenar(lote):
        for p in m.parameters():
            p.data = -1
    
    engine.entrenar_con_replay_con_rollback(m, entrenar, evaluar, n_replay=1)
    
    assert [p.data for p in m.parameters()] == pesos_originales


def test_rollback_en_error_entrenamiento(engine, modelo):
    """Si entrenar() lanza excepción, debe hacer rollback"""
    pesos_originales = [p.data for p in modelo.parameters()]
    
    scores = iter([0.9, 0.95])
    def evaluar(): return next(scores)
    
    def entrenar(lote):
        modelo.parameters()[0].data = 999.0
        raise RuntimeError("boom")
    
    r = engine.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert r["ok"] == False
    assert "Error" in r["razon"]
    assert r["rollback"] == True
    assert [p.data for p in modelo.parameters()] == pesos_originales


def test_rollback_en_error_evaluacion(engine, modelo):
    """Si evaluar() lanza excepción después de entrenar, debe hacer rollback"""
    pesos_originales = [p.data for p in modelo.parameters()]
    
    llamadas = []
    def evaluar():
        llamadas.append(1)
        if len(llamadas) == 1:
            return 0.9
        raise RuntimeError("eval boom")
    
    def entrenar(lote):
        modelo.parameters()[0].data = 999.0
    
    r = engine.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert r["ok"] == False
    assert "Error evaluando" in r["razon"]
    assert r["rollback"] == True
    assert [p.data for p in modelo.parameters()] == pesos_originales


# ============================================
# TESTS DE BUFFER VACÍO
# ============================================

def test_sin_experiencias_no_entrena(modelo):
    engine_vacio = LearningEngine(umbral_mejora=0.0)
    
    def evaluar(): return 0.5
    def entrenar(lote):
        modelo.parameters()[0].data = 999.0
    
    r = engine_vacio.entrenar_con_replay_con_rollback(
        modelo, entrenar, evaluar, n_replay=1
    )
    
    assert r["ok"] == False
    assert "vacío" in r["razon"].lower()
    # El modelo NO debe haberse modificado
    assert modelo.parameters()[0].data == 1.0


# ============================================
# TESTS DE INTEGRACIÓN
# ============================================

def test_historial_evaluaciones_crece(engine, modelo):
    scores = iter([0.5, 0.9, 0.9, 0.1])
    def evaluar(): return next(scores)
    def entrenar(lote): pass
    
    n_antes = len(engine.historial_evaluaciones)
    
    engine.entrenar_con_replay_con_rollback(modelo, entrenar, evaluar, n_replay=1)
    engine.entrenar_con_replay_con_rollback(modelo, entrenar, evaluar, n_replay=1)
    
    assert len(engine.historial_evaluaciones) == n_antes + 2


def test_flujo_completo_commit_then_rollback(engine):
    """Commit primero, rollback después"""
    m = MockModel([1.0, 2.0])
    
    # 1. Commit
    scores1 = iter([0.3, 0.9])
    def eval1(): return next(scores1)
    def train1(lote):
        m.parameters()[0].data = 10.0
        m.parameters()[1].data = 20.0
    
    r1 = engine.entrenar_con_replay_con_rollback(m, train1, eval1, n_replay=1)
    assert r1["ok"] == True
    assert m.parameters()[0].data == 10.0
    
    # 2. Rollback
    scores2 = iter([0.9, 0.1])
    def eval2(): return next(scores2)
    def train2(lote):
        m.parameters()[0].data = 999.0
        m.parameters()[1].data = 999.0
    
    r2 = engine.entrenar_con_replay_con_rollback(m, train2, eval2, n_replay=1)
    assert r2["ok"] == False
    # El modelo vuelve al estado del COMMIT anterior
    assert m.parameters()[0].data == 10.0
    assert m.parameters()[1].data == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
