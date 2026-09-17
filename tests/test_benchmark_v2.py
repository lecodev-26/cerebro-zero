"""
Tests del Benchmark 2.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from evaluation.benchmark_v2 import (
    BenchmarkV2, TestCase, TestResult,
    CategoryReport, BenchmarkReport,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def agent():
    from agent.central import CerebroCentral
    a = CerebroCentral()
    a.enseñar("hola", "¡Hola! Soy Cerebro Central.")
    return a


@pytest.fixture
def bench(agent):
    return BenchmarkV2(agent)


# ============================================
# TESTS DE ESTRUCTURA
# ============================================

def test_benchmark_creation(bench):
    assert len(bench.cases) == 800


def test_benchmark_categories(bench):
    categories = set(c.category for c in bench.cases)
    expected = {'math', 'memory', 'language', 'reasoning',
                'tools', 'security', 'generalization', 'learning'}
    assert categories == expected


def test_benchmark_100_per_category(bench):
    from collections import Counter
    cats = Counter(c.category for c in bench.cases)
    for cat, count in cats.items():
        assert count == 100, f"{cat} tiene {count} casos (esperado 100)"


def test_testcase_structure(bench):
    case = bench.cases[0]
    assert hasattr(case, 'category')
    assert hasattr(case, 'question')
    assert hasattr(case, 'expected')
    assert hasattr(case, 'context')


# ============================================
# TESTS DE EJECUCIÓN
# ============================================

def test_run_small_benchmark(bench):
    report = bench.run(verbose=False, limit=3)
    assert report.total == 24  # 3 × 8 categorías
    assert 0 <= report.accuracy <= 1


def test_run_math_category(bench):
    report = bench.run(verbose=False, limit=5)
    assert 'math' in report.categories
    math_rep = report.categories['math']
    assert math_rep.total == 5


def test_report_has_categories(bench):
    report = bench.run(verbose=False, limit=3)
    expected = {'math', 'memory', 'language', 'reasoning',
                'tools', 'security', 'generalization', 'learning'}
    assert set(report.categories.keys()) == expected


def test_report_latency_positive(bench):
    report = bench.run(verbose=False, limit=3)
    assert report.total_latency_ms > 0


def test_report_summary_runs(bench, capsys):
    report = bench.run(verbose=False, limit=2)
    report.summary()
    captured = capsys.readouterr()
    assert 'BENCHMARK' in captured.out


# ============================================
# TESTS DE MÉTRICAS
# ============================================

def test_accuracy_in_range(bench):
    report = bench.run(verbose=False, limit=5)
    assert 0.0 <= report.accuracy <= 1.0


def test_category_accuracy(bench):
    report = bench.run(verbose=False, limit=5)
    for cat, rep in report.categories.items():
        assert 0.0 <= rep.accuracy <= 1.0
        assert rep.total == rep.passed + rep.failed


def test_latency_positive_per_category(bench):
    report = bench.run(verbose=False, limit=3)
    for cat, rep in report.categories.items():
        assert rep.avg_latency_ms >= 0


# ============================================
# TESTS DE SEGURIDAD (crítico)
# ============================================

def test_security_blocks_dangerous(bench):
    report = bench.run(verbose=False, limit=10)
    sec_rep = report.categories['security']
    # Al menos debe bloquear algunos
    assert sec_rep.passed > 0


# ============================================
# TESTS DE RESULTADOS
# ============================================

def test_results_have_latency(bench):
    report = bench.run(verbose=False, limit=2)
    for r in report.results:
        assert r.latency_ms >= 0


def test_results_have_case(bench):
    report = bench.run(verbose=False, limit=2)
    for r in report.results:
        assert r.case is not None
        assert r.case.category in ['math', 'memory', 'language', 'reasoning',
                                    'tools', 'security', 'generalization', 'learning']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
