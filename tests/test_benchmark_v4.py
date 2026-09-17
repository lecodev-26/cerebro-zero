"""
Tests de Benchmark 4.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from evaluation.benchmark_v4 import (
    CasoBenchmark, ResultadoCaso, Metricas,
    GeneradorCasos, Benchmark, Reporte,
    handler_math, handler_memory, handler_planning,
    handler_rl, handler_learning,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def generador():
    return GeneradorCasos(seed=42)


@pytest.fixture
def bench_completo():
    bench = Benchmark(seed=42)
    bench.registrar_handler("math", handler_math)
    bench.registrar_handler("memory", handler_memory)
    bench.registrar_handler("planning", handler_planning)
    bench.registrar_handler("rl", handler_rl)
    bench.registrar_handler("learning", handler_learning)
    bench.cargar_casos()
    return bench


# ============================================
# TESTS DE CASO
# ============================================

def test_caso_creation():
    c = CasoBenchmark(id="x", categoria="math", split="KNOWN",
                      input="1+1", esperado=2)
    assert c.id == "x"
    assert c.categoria == "math"
    assert c.split == "KNOWN"


def test_caso_to_dict():
    c = CasoBenchmark(id="x", categoria="math", split="KNOWN",
                      input="1+1", esperado=2)
    d = c.to_dict()
    assert d["id"] == "x"
    assert d["esperado"] == "2"


# ============================================
# TESTS DE MÉTRICAS
# ============================================

def test_metricas_vacias():
    m = Metricas()
    assert m.accuracy == 0.0
    assert m.latency_p50 == 0.0


def test_metricas_accuracy():
    m = Metricas(total=10, correctos=7)
    assert m.accuracy == 0.7


def test_metricas_exact_match():
    m = Metricas(total=10, exact_match=5)
    assert m.exact_match_rate == 0.5


def test_metricas_error_rate():
    m = Metricas(total=10, errores=2)
    assert m.error_rate == 0.2


def test_metricas_latency_p50():
    m = Metricas(latencias=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert m.latency_p50 == 0.3


def test_metricas_latency_p95():
    m = Metricas(latencias=list(range(100)))
    p95 = m.latency_p95
    assert 94 <= p95 <= 96


def test_metricas_latency_media():
    m = Metricas(latencias=[0.1, 0.2, 0.3])
    assert m.latency_media == pytest.approx(0.2)


def test_metricas_to_dict():
    m = Metricas(total=10, correctos=7, latencias=[0.1])
    d = m.to_dict()
    assert "accuracy" in d
    assert d["accuracy"] == 0.7


# ============================================
# TESTS DE GENERADOR
# ============================================

def test_generador_math_known(generador):
    casos = generador.math_known(5)
    assert len(casos) == 5
    assert all(c.categoria == "math" for c in casos)
    assert all(c.split == "KNOWN" for c in casos)


def test_generador_math_unseen(generador):
    casos = generador.math_unseen(5)
    assert all(c.split == "UNSEEN" for c in casos)


def test_generador_math_adversarial(generador):
    casos = generador.math_adversarial(5)
    assert all(c.split == "ADVERSARIAL" for c in casos)


def test_generador_memory_known(generador):
    casos = generador.memory_known(5)
    assert len(casos) == 5


def test_generador_planning_known(generador):
    casos = generador.planning_known(5)
    assert all(c.categoria == "planning" for c in casos)
    assert all(c.esperado == "entrenar" for c in casos)


def test_generador_rl_known(generador):
    casos = generador.rl_known(3)
    assert all(c.categoria == "rl" for c in casos)


def test_generador_learning_known(generador):
    casos = generador.learning_known(5)
    assert all(c.categoria == "learning" for c in casos)


def test_generador_todos(generador):
    casos = generador.generar_todos()
    assert len(casos) > 100
    cats = set(c.categoria for c in casos)
    assert "math" in cats
    assert "memory" in cats
    assert "planning" in cats
    assert "rl" in cats
    assert "learning" in cats


def test_generador_splits_presentes(generador):
    casos = generador.generar_todos()
    splits = set(c.split for c in casos)
    assert "KNOWN" in splits
    assert "UNSEEN" in splits
    assert "ADVERSARIAL" in splits


def test_generador_reproducible():
    g1 = GeneradorCasos(seed=42)
    g2 = GeneradorCasos(seed=42)
    casos1 = g1.math_known(5)
    casos2 = g2.math_known(5)
    for c1, c2 in zip(casos1, casos2):
        assert c1.input == c2.input
        assert c1.esperado == c2.esperado


# ============================================
# TESTS DE BENCHMARK
# ============================================

def test_benchmark_creation():
    b = Benchmark(seed=42)
    assert b.seed == 42
    assert b.casos == []


def test_benchmark_registrar_handler():
    b = Benchmark()
    b.registrar_handler("math", handler_math)
    assert "math" in b.handlers


def test_benchmark_cargar_casos(bench_completo):
    assert len(bench_completo.casos) > 0


def test_benchmark_ejecutar_caso_ok():
    b = Benchmark()
    b.registrar_handler("math", handler_math)
    caso = CasoBenchmark(id="x", categoria="math", split="KNOWN",
                          input="2 + 3", esperado=5)
    r = b.ejecutar_caso(caso)
    assert r.correcto == True
    assert r.output == 5
    assert r.exact_match == True


def test_benchmark_ejecutar_caso_sin_handler():
    b = Benchmark()
    caso = CasoBenchmark(id="x", categoria="unknown", split="KNOWN",
                          input="x", esperado=1)
    r = b.ejecutar_caso(caso)
    assert r.correcto == False
    assert r.error is not None


def test_benchmark_ejecutar_caso_error_handler():
    b = Benchmark()
    def handler_malo(caso):
        raise ValueError("boom")
    b.registrar_handler("math", handler_malo)
    caso = CasoBenchmark(id="x", categoria="math", split="KNOWN",
                          input="x", esperado=1)
    r = b.ejecutar_caso(caso)
    assert r.correcto == False
    assert "boom" in r.error


def test_benchmark_comparar_int():
    b = Benchmark()
    assert b._comparar(5, 5) == True
    assert b._comparar(5, 6) == False
    assert b._comparar(5.0, 5) == True


def test_benchmark_comparar_none():
    b = Benchmark()
    assert b._comparar(None, None) == True
    assert b._comparar("", None) == True
    assert b._comparar("x", None) == False


def test_benchmark_comparar_string():
    b = Benchmark()
    assert b._comparar("Hola Mundo", "hola") == True
    assert b._comparar("Hola", "adios") == False


def test_benchmark_ejecutar_completo(bench_completo):
    reporte = bench_completo.ejecutar()
    assert reporte is not None
    assert len(reporte.resultados) == len(bench_completo.casos)


def test_benchmark_repr():
    b = Benchmark()
    assert "Benchmark" in repr(b)


# ============================================
# TESTS DE REPORTE
# ============================================

def test_reporte_global_metrics(bench_completo):
    reporte = bench_completo.ejecutar()
    g = reporte.global_metrics()
    assert g.total == len(bench_completo.casos)
    assert 0 <= g.accuracy <= 1


def test_reporte_por_categoria(bench_completo):
    reporte = bench_completo.ejecutar()
    por_cat = reporte.por_categoria()
    assert "math" in por_cat
    assert "memory" in por_cat
    assert "planning" in por_cat


def test_reporte_por_split(bench_completo):
    reporte = bench_completo.ejecutar()
    por_split = reporte.por_split()
    assert "KNOWN" in por_split
    assert "UNSEEN" in por_split
    assert "ADVERSARIAL" in por_split


def test_reporte_por_categoria_y_split(bench_completo):
    reporte = bench_completo.ejecutar()
    doble = reporte.por_categoria_y_split()
    assert "math" in doble
    assert "KNOWN" in doble["math"]


def test_reporte_to_dict(bench_completo):
    reporte = bench_completo.ejecutar()
    d = reporte.to_dict()
    assert "global" in d
    assert "por_categoria" in d
    assert "por_split" in d


def test_reporte_guardar(bench_completo):
    reporte = bench_completo.ejecutar()
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        reporte.guardar(tmp)
        assert os.path.exists(tmp)
        import json
        with open(tmp) as f:
            data = json.load(f)
        assert "global" in data
    finally:
        os.unlink(tmp)


def test_reporte_resumen_texto(bench_completo):
    reporte = bench_completo.ejecutar()
    texto = reporte.resumen_texto()
    assert "REPORTE BENCHMARK" in texto
    assert "GLOBAL" in texto


def test_reporte_repr(bench_completo):
    reporte = bench_completo.ejecutar()
    assert "Reporte" in repr(reporte)


# ============================================
# TESTS DE RENDIMIENTO
# ============================================

def test_benchmark_math_alto(bench_completo):
    """El handler de math debe ser casi perfecto"""
    reporte = bench_completo.ejecutar()
    por_cat = reporte.por_categoria()
    assert por_cat["math"].accuracy >= 0.95


def test_benchmark_memory_alto(bench_completo):
    reporte = bench_completo.ejecutar()
    por_cat = reporte.por_categoria()
    assert por_cat["memory"].accuracy >= 0.95


def test_benchmark_planning_alto(bench_completo):
    reporte = bench_completo.ejecutar()
    por_cat = reporte.por_categoria()
    assert por_cat["planning"].accuracy >= 0.90


def test_benchmark_adversarial_decente(bench_completo):
    """Adversarial debe ser >50% con handlers correctos"""
    reporte = bench_completo.ejecutar()
    por_split = reporte.por_split()
    assert por_split["ADVERSARIAL"].accuracy >= 0.5


def test_benchmark_latencias_positivas(bench_completo):
    reporte = bench_completo.ejecutar()
    for r in reporte.resultados:
        assert r.latencia >= 0.0


# ============================================
# TESTS DE HANDLERS INDIVIDUALES
# ============================================

def test_handler_math_suma():
    caso = CasoBenchmark(id="x", categoria="math", split="KNOWN",
                          input="5 + 3", esperado=8)
    assert handler_math(caso) == 8


def test_handler_math_error():
    caso = CasoBenchmark(id="x", categoria="math", split="KNOWN",
                          input="basura", esperado=None)
    assert handler_math(caso) is None


def test_handler_memory_recuerda():
    caso = CasoBenchmark(id="x", categoria="memory", split="KNOWN",
                          input="recuerda color = azul",
                          esperado="color = azul")
    out = handler_memory(caso)
    assert "color = azul" in out


def test_handler_memory_no_existe():
    caso = CasoBenchmark(id="x", categoria="memory", split="ADVERSARIAL",
                          input="recordar no_existe_0", esperado=None)
    assert handler_memory(caso) is None


def test_handler_planning_entrenar():
    caso = CasoBenchmark(id="x", categoria="planning", split="KNOWN",
                          input="Entrenar el modelo", esperado="entrenar")
    assert handler_planning(caso) == "entrenar"


def test_handler_learning_rechazar():
    caso = CasoBenchmark(id="x", categoria="learning", split="ADVERSARIAL",
                          input="actualizar con score 0.1 sobre 0.9",
                          esperado="rechazar")
    assert handler_learning(caso) == "rechazar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
