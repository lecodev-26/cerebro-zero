"""End-to-end benchmark for the real CerebroV3 pipeline.

Phase 4.4: measures the integrated agent instead of isolated handlers.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.cerebro_v3 import CerebroV3


@dataclass
class E2ECase:
    id: str
    category: str
    input: str
    expected: str
    setup: list[str] = field(default_factory=list)


@dataclass
class E2EResult:
    case: E2ECase
    output: str
    correct: bool
    latency: float
    tool: str | None
    confidence: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "category": self.case.category,
            "input": self.case.input,
            "expected": self.case.expected,
            "output": self.output,
            "correct": self.correct,
            "latency_ms": round(self.latency * 1000, 3),
            "tool": self.tool,
            "confidence": self.confidence,
            "error": self.error,
        }


class E2EBenchmark:
    """Benchmark the actual CerebroV3 public processing pipeline."""

    def __init__(self, seed: int = 42, factory: Callable[..., CerebroV3] = CerebroV3):
        self.seed = seed
        self.factory = factory
        self.cases = self.default_cases()
        self.results: list[E2EResult] = []

    @staticmethod
    def default_cases() -> list[E2ECase]:
        return [
            E2ECase("math_add", "math", "5 + 3", "8.0"),
            E2ECase("math_mul", "math", "4 * 5", "20.0"),
            E2ECase("learn", "memory", "aprende color = azul", "azul"),
            E2ECase("recall", "memory", "recordar color", "azul"),
            E2ECase("plan", "planning", "Entrenar el modelo", "Plan"),
            E2ECase("unknown", "robustness", "xyzzy foobar", "No entiendo"),
        ]

    def run(self) -> list[E2EResult]:
        cerebro = self.factory(nombre="e2e", verbose=False)
        self.results = []
        for case in self.cases:
            try:
                for setup in case.setup:
                    cerebro.procesar(setup)
                t0 = time.perf_counter()
                response = cerebro.procesar(case.input)
                latency = time.perf_counter() - t0
                output = str(response.output)
                self.results.append(E2EResult(
                    case=case,
                    output=output,
                    correct=case.expected.lower() in output.lower(),
                    latency=latency,
                    tool=response.tool_usada,
                    confidence=response.confianza,
                ))
            except Exception as exc:
                self.results.append(E2EResult(
                    case=case, output="", correct=False, latency=0.0,
                    tool=None, confidence=0.0, error=str(exc),
                ))
        return self.results

    def summary(self) -> dict[str, Any]:
        if not self.results:
            self.run()
        total = len(self.results)
        correct = sum(r.correct for r in self.results)
        latencies = sorted(r.latency for r in self.results)
        p95 = latencies[min(int(total * 0.95), total - 1)] if total else 0.0
        return {
            "benchmark": "cerebro-v4-e2e",
            "seed": self.seed,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "latency_p50_ms": (latencies[total // 2] * 1000) if total else 0.0,
            "latency_p95_ms": p95 * 1000,
            "errors": sum(r.error is not None for r in self.results),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    benchmark = E2EBenchmark()
    benchmark.run()
    print(json.dumps(benchmark.summary(), indent=2, ensure_ascii=False))
