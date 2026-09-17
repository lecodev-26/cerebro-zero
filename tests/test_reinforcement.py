"""
Tests de RL Básico (GridWorld + Q-Learning)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import random
from training.reinforcement import (
    GridWorld, QLearning, Episodio, EntrenadorRL,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def env_simple():
    return GridWorld(filas=3, columnas=3, inicio=(0, 0), meta=(2, 2))


@pytest.fixture
def env_5x5():
    return GridWorld(filas=5, columnas=5)


@pytest.fixture
def agente():
    return QLearning(num_estados=9, num_acciones=4)


@pytest.fixture
def entrenador(env_5x5):
    random.seed(42)
    return EntrenadorRL(env_5x5, epsilon_decay=0.99)


# ============================================
# TESTS DE GRIDWORLD
# ============================================

def test_gridworld_creation(env_simple):
    assert env_simple.filas == 3
    assert env_simple.columnas == 3
    assert env_simple.inicio == (0, 0)
    assert env_simple.meta == (2, 2)


def test_gridworld_num_estados(env_simple):
    assert env_simple.num_estados() == 9


def test_gridworld_num_acciones(env_simple):
    assert env_simple.num_acciones() == 4


def test_gridworld_reset(env_simple):
    env_simple.step(3)
    estado = env_simple.reset()
    assert estado == (0, 0)
    assert env_simple.pasos == 0
    assert env_simple.recompensa_acumulada == 0.0


def test_gridworld_step_derecha(env_simple):
    env_simple.reset()
    estado, r, term, info = env_simple.step(3)
    assert estado == (0, 1)
    assert r == -0.1
    assert term == False


def test_gridworld_step_abajo(env_simple):
    env_simple.reset()
    estado, r, term, info = env_simple.step(1)
    assert estado == (1, 0)


def test_gridworld_step_limites(env_simple):
    env_simple.reset()
    # Arriba desde (0,0) no debe salir
    estado, r, term, info = env_simple.step(0)
    assert estado == (0, 0)


def test_gridworld_step_meta(env_simple):
    env_simple.reset()
    # Ir a (0,2), luego abajo x2
    env_simple.step(3)
    env_simple.step(3)
    env_simple.step(1)
    estado, r, term, info = env_simple.step(1)
    assert estado == (2, 2)
    assert r == 10.0
    assert term == True


def test_gridworld_obstaculo():
    env = GridWorld(filas=3, columnas=3, inicio=(0, 0), meta=(2, 2),
                    obstaculos=[(0, 1)])
    env.reset()
    estado, r, term, info = env.step(3)  # derecha → choca
    assert r == -5.0
    assert estado == (0, 0)  # reset a inicio


def test_gridworld_max_pasos():
    env = GridWorld(filas=10, columnas=10, inicio=(0, 0), meta=(9, 9),
                    max_pasos=5)
    env.reset()
    terminado = False
    for _ in range(10):
        _, _, terminado, _ = env.step(3)
        if terminado:
            break
    assert terminado == True


def test_gridworld_estado_a_indice(env_simple):
    assert env_simple.estado_a_indice((0, 0)) == 0
    assert env_simple.estado_a_indice((0, 1)) == 1
    assert env_simple.estado_a_indice((1, 0)) == 3


def test_gridworld_indice_a_estado(env_simple):
    assert env_simple.indice_a_estado(0) == (0, 0)
    assert env_simple.indice_a_estado(4) == (1, 1)


def test_gridworld_render(env_simple):
    env_simple.reset()
    render = env_simple.render()
    assert "A" in render
    assert "🎯" in render


def test_gridworld_repr(env_simple):
    assert "GridWorld" in repr(env_simple)


def test_gridworld_meta_no_en_obstaculos():
    env = GridWorld(filas=3, columnas=3, meta=(1, 1),
                    obstaculos=[(1, 1)])
    assert (1, 1) not in env.obstaculos


# ============================================
# TESTS DE Q-LEARNING
# ============================================

def test_qlearning_creation(agente):
    assert agente.num_estados == 9
    assert agente.num_acciones == 4
    assert agente.alpha == 0.1
    assert agente.gamma == 0.95


def test_qlearning_q_inicial_cero(agente):
    for s in range(agente.num_estados):
        for a in range(agente.num_acciones):
            assert agente.Q[s][a] == 0.0


def test_qlearning_mejor_accion_inicial(agente):
    # Todos 0 → cualquiera vale
    a = agente.mejor_accion(0)
    assert 0 <= a < 4


def test_qlearning_elegir_accion_exploracion(agente):
    random.seed(42)
    agente.epsilon = 1.0  # siempre explora
    acciones = [agente.elegir_accion(0) for _ in range(50)]
    # Debe haber variedad
    assert len(set(acciones)) > 1


def test_qlearning_elegir_accion_explotacion(agente):
    agente.epsilon = 0.0
    # Establecer Q[0][2] como mejor
    agente.Q[0][2] = 100.0
    a = agente.elegir_accion(0)
    assert a == 2


def test_qlearning_aprender(agente):
    agente.aprender(0, 0, 1.0, 1, False)
    # Q[0][0] = 0 + 0.1 * (1.0 + 0.95*0 - 0) = 0.1
    assert agente.Q[0][0] == pytest.approx(0.1)


def test_qlearning_aprender_terminado(agente):
    agente.aprender(0, 0, 10.0, 1, True)
    # target = 10.0, Q = 0 + 0.1 * (10 - 0) = 1.0
    assert agente.Q[0][0] == pytest.approx(1.0)


def test_qlearning_valor_estado(agente):
    agente.Q[0] = [0.0, 5.0, 3.0, 1.0]
    assert agente.valor_estado(0) == 5.0


def test_qlearning_politica(agente):
    agente.Q[0][2] = 100.0
    agente.Q[1][3] = 50.0
    politica = agente.politica()
    assert politica[0] == 2
    assert politica[1] == 3


def test_qlearning_reset(agente):
    agente.aprender(0, 0, 1.0, 1, False)
    agente.reset()
    assert agente.Q[0][0] == 0.0


def test_qlearning_repr(agente):
    assert "QLearning" in repr(agente)


# ============================================
# TESTS DE EPISODIO
# ============================================

def test_episodio_creation():
    ep = Episodio(num=1, pasos=10, recompensa_total=5.0,
                  exito=True, epsilon=0.5)
    assert ep.num == 1
    assert ep.pasos == 10
    assert ep.exito == True


def test_episodio_to_dict():
    ep = Episodio(num=1, pasos=10, recompensa_total=5.0,
                  exito=True, epsilon=0.5)
    d = ep.to_dict()
    assert d["num"] == 1
    assert d["exito"] == True


# ============================================
# TESTS DE ENTRENADOR RL
# ============================================

def test_entrenador_creation(env_5x5):
    entrenador = EntrenadorRL(env_5x5)
    assert entrenador.entorno == env_5x5
    assert entrenador.agente is not None
    assert len(entrenador.episodios) == 0


def test_entrenador_ejecutar_episodio(entrenador):
    ep = entrenador.ejecutar_episodio(0)
    assert ep.num == 0
    assert ep.pasos > 0
    assert len(entrenador.episodios) == 1


def test_entrenador_entrenar(entrenador):
    metricas = entrenador.entrenar(num_episodios=20)
    assert metricas["episodios"] == 20
    assert "tasa_exito" in metricas


def test_entrenador_entrenar_mejora(env_5x5):
    """
    Con entrenamiento, la tasa de éxito debe subir.
    Comparamos primeros vs últimos episodios.
    """
    random.seed(42)
    entrenador = EntrenadorRL(env_5x5, epsilon_decay=0.95)
    entrenador.entrenar(num_episodios=200)
    
    primeros = entrenador.episodios[:30]
    ultimos = entrenador.episodios[-30:]
    
    exitos_primeros = sum(1 for e in primeros if e.exito)
    exitos_ultimos = sum(1 for e in ultimos if e.exito)
    
    # Debe aprender: últimos mejor que primeros
    assert exitos_ultimos >= exitos_primeros


def test_entrenador_metricas_sin_episodios(entrenador):
    m = entrenador.metricas()
    assert m["episodios"] == 0


def test_entrenador_metricas_completas(entrenador):
    entrenador.entrenar(num_episodios=20)
    m = entrenador.metricas()
    assert "episodios" in m
    assert "exitos" in m
    assert "tasa_exito" in m
    assert "recompensa_media" in m
    assert "epsilon_final" in m


def test_entrenador_epsilon_decae(entrenador):
    eps_inicial = entrenador.agente.epsilon
    entrenador.entrenar(num_episodios=20)
    assert entrenador.agente.epsilon < eps_inicial


def test_entrenador_epsilon_min(env_5x5):
    entrenador = EntrenadorRL(env_5x5, epsilon_decay=0.5, epsilon_min=0.1)
    entrenador.entrenar(num_episodios=50)
    assert entrenador.agente.epsilon >= 0.1


def test_entrenador_evaluar(entrenador):
    entrenador.entrenar(num_episodios=100)
    eval_metrics = entrenador.evaluar(num_episodios=10)
    assert eval_metrics["episodios"] == 10
    assert 0.0 <= eval_metrics["tasa_exito"] <= 1.0


def test_entrenador_evaluar_restaura_epsilon(entrenador):
    entrenador.entrenar(num_episodios=10)
    eps_antes = entrenador.agente.epsilon
    entrenador.evaluar(num_episodios=5)
    assert entrenador.agente.epsilon == eps_antes


def test_entrenador_guardar(entrenador):
    entrenador.entrenar(num_episodios=10)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        entrenador.guardar(tmp)
        assert os.path.exists(tmp)
        
        import json
        with open(tmp) as f:
            data = json.load(f)
        
        assert "Q" in data
        assert "metricas" in data
        assert len(data["Q"]) == 25  # 5x5
    finally:
        os.unlink(tmp)


def test_entrenador_repr(entrenador):
    assert "EntrenadorRL" in repr(entrenador)


# ============================================
# TESTS DE APRENDIZAJE REAL
# ============================================

def test_qlearning_aprende_camino_corto():
    """
    En un GridWorld 3x3 desde (0,0) a (2,2), el camino óptimo son 4 pasos.
    Tras entrenamiento, el agente debe hacer 4 pasos la mayoría de las veces.
    """
    random.seed(42)
    env = GridWorld(filas=3, columnas=3)
    entrenador = EntrenadorRL(env, epsilon_decay=0.95)
    entrenador.entrenar(num_episodios=300)
    
    eval_metrics = entrenador.evaluar(num_episodios=20)
    # Con 3x3 el camino óptimo son 4 pasos
    # Debe estar cerca del óptimo (< 8 pasos de media)
    assert eval_metrics["pasos_medios"] <= 8


def test_gridworld_con_obstaculos():
    """GridWorld con obstáculos debe seguir siendo entrenable"""
    random.seed(42)
    env = GridWorld(filas=4, columnas=4, inicio=(0, 0), meta=(3, 3),
                    obstaculos=[(1, 1), (1, 2), (2, 1)])
    entrenador = EntrenadorRL(env, epsilon_decay=0.95)
    entrenador.entrenar(num_episodios=200)
    
    eval_metrics = entrenador.evaluar(num_episodios=10)
    # Debe encontrar camino
    assert eval_metrics["tasa_exito"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
