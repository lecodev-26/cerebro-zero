"""Tests for the real CerebroV3 end-to-end benchmark."""

from evaluation.benchmark_e2e import E2EBenchmark


def test_e2e_cases_are_realistic():
    cases = E2EBenchmark.default_cases()
    assert len(cases) >= 5
    assert any(c.category == "math" for c in cases)
    assert any(c.category == "memory" for c in cases)
    assert any(c.category == "planning" for c in cases)


def test_e2e_benchmark_runs_real_pipeline():
    benchmark = E2EBenchmark(seed=42)
    results = benchmark.run()
    assert len(results) == len(benchmark.cases)
    assert all(r.output or r.error for r in results)
    assert sum(r.correct for r in results) >= 5


def test_e2e_summary_is_reproducible_shape():
    summary = E2EBenchmark(seed=42).summary()
    assert summary["benchmark"] == "cerebro-v4-e2e"
    assert summary["total"] >= 5
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert "results" in summary


def test_e2e_memory_is_stateful():
    benchmark = E2EBenchmark(seed=42)
    benchmark.run()
    recall = next(r for r in benchmark.results if r.case.id == "recall")
    assert recall.correct
    assert recall.tool == "recordar"
