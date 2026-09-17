"""
Benchmark 4.0 — Benchmark científico con splits y métricas reales
====================================================================

Mejoras sobre benchmark_v3:
- Categorías nuevas: planning, rl, learning
- Métricas científicas: accuracy, exact_match, latency_p50/p95, error_rate
- Splits: KNOWN, UNSEEN, ADVERSARIAL
- Resultados reproducibles con seed
- Categorías separadas por tipo de fenómeno cognitivo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime


# ============================================
# ESTRUCTURAS
# ============================================

@dataclass
class CasoBenchmark:
    """Un caso individual de benchmark"""
    id: str
    categoria: str
    split: str  # KNOWN, UNSEEN, ADVERSARIAL
    input: str
    esperado: Any
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "categoria": self.categoria,
            "split": self.split,
            "input": self.input,
            "esperado": str(self.esperado)[:200],
            "metadata": self.metadata,
        }


@dataclass
class ResultadoCaso:
    """Resultado de ejecutar un caso"""
    caso: CasoBenchmark
    output: Any
    correcto: bool
    exact_match: bool
    latencia: float
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.caso.id,
            "categoria": self.caso.categoria,
            "split": self.caso.split,
            "correcto": self.correcto,
            "exact_match": self.exact_match,
            "latencia": self.latencia,
            "error": self.error,
        }


@dataclass
class Metricas:
    """Métricas agregadas de un conjunto de resultados"""
    total: int = 0
    correctos: int = 0
    exact_match: int = 0
    errores: int = 0
    latencias: List[float] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        return self.correctos / self.total if self.total > 0 else 0.0
    
    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / self.total if self.total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.errores / self.total if self.total > 0 else 0.0
    
    @property
    def latency_p50(self) -> float:
        if not self.latencias:
            return 0.0
        return statistics.median(self.latencias)
    
    @property
    def latency_p95(self) -> float:
        if not self.latencias:
            return 0.0
        sorted_lat = sorted(self.latencias)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def latency_media(self) -> float:
        return sum(self.latencias) / len(self.latencias) if self.latencias else 0.0
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correctos": self.correctos,
            "accuracy": round(self.accuracy, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "latency_p50": round(self.latency_p50, 4),
            "latency_p95": round(self.latency_p95, 4),
            "latency_media": round(self.latency_media, 4),
        }


# ============================================
# GENERADORES DE CASOS POR CATEGORÍA
# ============================================

class GeneradorCasos:
    """Genera casos de benchmark por categoría y split"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
    
    # ============================================
    # MATH
    # ============================================
    
    def math_known(self, n: int = 20) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            a = self.rng.randint(1, 10)
            b = self.rng.randint(1, 10)
            casos.append(CasoBenchmark(
                id=f"math_known_{i}",
                categoria="math",
                split="KNOWN",
                input=f"{a} + {b}",
                esperado=a + b,
                metadata={"a": a, "b": b, "op": "+"},
            ))
        return casos
    
    def math_unseen(self, n: int = 20) -> List[CasoBenchmark]:
        """Números más grandes que en KNOWN"""
        casos = []
        for i in range(n):
            a = self.rng.randint(50, 200)
            b = self.rng.randint(50, 200)
            casos.append(CasoBenchmark(
                id=f"math_unseen_{i}",
                categoria="math",
                split="UNSEEN",
                input=f"{a} + {b}",
                esperado=a + b,
                metadata={"a": a, "b": b, "op": "+"},
            ))
        return casos
    
    def math_adversarial(self, n: int = 10) -> List[CasoBenchmark]:
        """Casos con trampa"""
        casos = []
        trampas = [
            ("0 + 0", 0),
            ("1 + -1", 0),
            ("100 + 0", 100),
        ]
        for i in range(n):
            inp, esp = trampas[i % len(trampas)]
            casos.append(CasoBenchmark(
                id=f"math_adv_{i}",
                categoria="math",
                split="ADVERSARIAL",
                input=inp,
                esperado=esp,
            ))
        return casos
    
    # ============================================
    # MEMORY
    # ============================================
    
    def memory_known(self, n: int = 20) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"memory_known_{i}",
                categoria="memory",
                split="KNOWN",
                input=f"recuerda clave_{i} = valor_{i}",
                esperado=f"clave_{i} = valor_{i}",
                metadata={"clave": f"clave_{i}", "valor": f"valor_{i}"},
            ))
        return casos
    
    def memory_unseen(self, n: int = 20) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"memory_unseen_{i}",
                categoria="memory",
                split="UNSEEN",
                input=f"recuerda color_{i} = rgb_{i}",
                esperado=f"color_{i} = rgb_{i}",
            ))
        return casos
    
    def memory_adversarial(self, n: int = 5) -> List[CasoBenchmark]:
        """Recuperar algo que no existe"""
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"memory_adv_{i}",
                categoria="memory",
                split="ADVERSARIAL",
                input=f"recordar no_existe_{i}",
                esperado=None,  # No debe inventarse nada
            ))
        return casos
    
    # ============================================
    # PLANNING (nueva)
    # ============================================
    
    def planning_known(self, n: int = 10) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"planning_known_{i}",
                categoria="planning",
                split="KNOWN",
                input="Entrenar el modelo",
                esperado="entrenar",  # intención detectada
            ))
        return casos
    
    def planning_unseen(self, n: int = 10) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"planning_unseen_{i}",
                categoria="planning",
                split="UNSEEN",
                input="evaluar el modelo en test",
                esperado="evaluar",
            ))
        return casos
    
    def planning_adversarial(self, n: int = 5) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"planning_adv_{i}",
                categoria="planning",
                split="ADVERSARIAL",
                input="xyzzy foobar",
                esperado=None,  # no reconocible → genérico
            ))
        return casos
    
    # ============================================
    # RL (nueva)
    # ============================================
    
    def rl_known(self, n: int = 5) -> List[CasoBenchmark]:
        """Casos de RL: verificar que el entorno funciona"""
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"rl_known_{i}",
                categoria="rl",
                split="KNOWN",
                input=f"gridworld_3x3_step_{i}",
                esperado="step_ok",
            ))
        return casos
    
    def rl_unseen(self, n: int = 5) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"rl_unseen_{i}",
                categoria="rl",
                split="UNSEEN",
                input=f"gridworld_5x5_train_{i}",
                esperado="entrena_ok",
            ))
        return casos
    
    # ============================================
    # LEARNING (nueva)
    # ============================================
    
    def learning_known(self, n: int = 10) -> List[CasoBenchmark]:
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"learning_known_{i}",
                categoria="learning",
                split="KNOWN",
                input=f"aceptar mejora {0.5 + i*0.05} sobre 0.5",
                esperado="aceptar" if i > 0 else "rechazar",
            ))
        return casos
    
    def learning_adversarial(self, n: int = 5) -> List[CasoBenchmark]:
        """Intentar actualizar con score PEOR → debe rechazar"""
        casos = []
        for i in range(n):
            casos.append(CasoBenchmark(
                id=f"learning_adv_{i}",
                categoria="learning",
                split="ADVERSARIAL",
                input=f"actualizar con score 0.1 sobre 0.9",
                esperado="rechazar",
            ))
        return casos
    
    # ============================================
    # GENERADOR COMPLETO
    # ============================================
    
    def generar_todos(self) -> List[CasoBenchmark]:
        casos = []
        casos.extend(self.math_known(20))
        casos.extend(self.math_unseen(20))
        casos.extend(self.math_adversarial(10))
        casos.extend(self.memory_known(20))
        casos.extend(self.memory_unseen(20))
        casos.extend(self.memory_adversarial(5))
        casos.extend(self.planning_known(10))
        casos.extend(self.planning_unseen(10))
        casos.extend(self.planning_adversarial(5))
        casos.extend(self.rl_known(5))
        casos.extend(self.rl_unseen(5))
        casos.extend(self.learning_known(10))
        casos.extend(self.learning_adversarial(5))
        return casos


# ============================================
# BENCHMARK RUNNER
# ============================================

class Benchmark:
    """
    Ejecuta casos y produce métricas.
    
    Uso:
        bench = Benchmark()
        bench.registrar_handler("math", mi_handler)
        reporte = bench.ejecutar()
    """
    
    def __init__(self, seed: int = 42, generador: Optional[GeneradorCasos] = None):
        self.seed = seed
        self.generador = generador or GeneradorCasos(seed=seed)
        self.casos: List[CasoBenchmark] = []
        self.handlers: Dict[str, Callable] = {}
        self.resultados: List[ResultadoCaso] = []
    
    def registrar_handler(self, categoria: str, handler: Callable[[CasoBenchmark], Any]):
        """
        Handler debe devolver el output del caso.
        Se comparará con caso.esperado.
        """
        self.handlers[categoria] = handler
    
    def cargar_casos(self, casos: Optional[List[CasoBenchmark]] = None):
        self.casos = casos if casos is not None else self.generador.generar_todos()
    
    def ejecutar_caso(self, caso: CasoBenchmark) -> ResultadoCaso:
        handler = self.handlers.get(caso.categoria)
        
        if handler is None:
            return ResultadoCaso(
                caso=caso,
                output=None,
                correcto=False,
                exact_match=False,
                latencia=0.0,
                error=f"No handler para categoría {caso.categoria}",
            )
        
        t0 = time.time()
        try:
            output = handler(caso)
            latencia = time.time() - t0
            correcto = self._comparar(output, caso.esperado)
            exact_match = (output == caso.esperado)
            return ResultadoCaso(
                caso=caso,
                output=output,
                correcto=correcto,
                exact_match=exact_match,
                latencia=latencia,
            )
        except Exception as e:
            latencia = time.time() - t0
            return ResultadoCaso(
                caso=caso,
                output=None,
                correcto=False,
                exact_match=False,
                latencia=latencia,
                error=str(e),
            )
    
    def _comparar(self, output: Any, esperado: Any) -> bool:
        """
        Comparación inteligente:
        - Si esperado es None → correcto si output es None o ""
        - Si es número → igualdad
        - Si es string → case-insensitive contiene
        """
        if esperado is None:
            return output is None or output == ""
        
        if isinstance(esperado, (int, float)):
            try:
                return abs(float(output) - float(esperado)) < 1e-6
            except (ValueError, TypeError):
                return False
        
        if isinstance(esperado, str):
            if output is None:
                return False
            return esperado.lower() in str(output).lower()
        
        return output == esperado
    
    def ejecutar(self) -> "Reporte":
        self.resultados = [self.ejecutar_caso(c) for c in self.casos]
        return Reporte(self.resultados, seed=self.seed)
    
    def __repr__(self):
        return f"Benchmark(casos={len(self.casos)}, handlers={list(self.handlers.keys())})"


# ============================================
# REPORTE
# ============================================

class Reporte:
    """Reporte de benchmark con métricas por categoría y split"""
    
    def __init__(self, resultados: List[ResultadoCaso], seed: int = 42):
        self.resultados = resultados
        self.seed = seed
        self.timestamp = time.time()
    
    def _agregar(self, resultados: List[ResultadoCaso]) -> Metricas:
        m = Metricas()
        for r in resultados:
            m.total += 1
            if r.correcto:
                m.correctos += 1
            if r.exact_match:
                m.exact_match += 1
            if r.error:
                m.errores += 1
            m.latencias.append(r.latencia)
        return m
    
    def global_metrics(self) -> Metricas:
        return self._agregar(self.resultados)
    
    def por_categoria(self) -> Dict[str, Metricas]:
        cats: Dict[str, List[ResultadoCaso]] = {}
        for r in self.resultados:
            cats.setdefault(r.caso.categoria, []).append(r)
        return {c: self._agregar(rs) for c, rs in cats.items()}
    
    def por_split(self) -> Dict[str, Metricas]:
        splits: Dict[str, List[ResultadoCaso]] = {}
        for r in self.resultados:
            splits.setdefault(r.caso.split, []).append(r)
        return {s: self._agregar(rs) for s, rs in splits.items()}
    
    def por_categoria_y_split(self) -> Dict[str, Dict[str, Metricas]]:
        result: Dict[str, Dict[str, List[ResultadoCaso]]] = {}
        for r in self.resultados:
            result.setdefault(r.caso.categoria, {}).setdefault(r.caso.split, [])
            result[r.caso.categoria][r.caso.split].append(r)
        
        return {
            cat: {split: self._agregar(rs) for split, rs in splits.items()}
            for cat, splits in result.items()
        }
    
    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "timestamp": self.timestamp,
            "global": self.global_metrics().to_dict(),
            "por_categoria": {
                c: m.to_dict() for c, m in self.por_categoria().items()
            },
            "por_split": {
                s: m.to_dict() for s, m in self.por_split().items()
            },
        }
    
    def guardar(self, archivo: str):
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        with open(archivo, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def resumen_texto(self) -> str:
        lineas = []
        lineas.append("=" * 60)
        lineas.append(f"📊 REPORTE BENCHMARK (seed={self.seed})")
        lineas.append("=" * 60)
        
        g = self.global_metrics()
        lineas.append(f"\n🌍 GLOBAL:")
        lineas.append(f"   Total: {g.total}")
        lineas.append(f"   Accuracy: {g.accuracy*100:.2f}%")
        lineas.append(f"   Exact match: {g.exact_match_rate*100:.2f}%")
        lineas.append(f"   Error rate: {g.error_rate*100:.2f}%")
        lineas.append(f"   Latency p50: {g.latency_p50*1000:.2f}ms")
        lineas.append(f"   Latency p95: {g.latency_p95*1000:.2f}ms")
        
        lineas.append(f"\n📂 POR CATEGORÍA:")
        for cat, m in sorted(self.por_categoria().items()):
            lineas.append(
                f"   {cat:12s}: {m.accuracy*100:6.2f}% "
                f"({m.correctos}/{m.total})  "
                f"p95={m.latency_p95*1000:.1f}ms"
            )
        
        lineas.append(f"\n🔀 POR SPLIT:")
        for split, m in sorted(self.por_split().items()):
            lineas.append(
                f"   {split:12s}: {m.accuracy*100:6.2f}% "
                f"({m.correctos}/{m.total})"
            )
        
        return "\n".join(lineas)
    
    def __repr__(self):
        g = self.global_metrics()
        return f"Reporte({g.total} casos, acc={g.accuracy*100:.1f}%)"


# ============================================
# HANDLERS DE EJEMPLO (para test manual)
# ============================================

def handler_math(caso: CasoBenchmark) -> Any:
    """Evalúa suma simple"""
    try:
        a, op, b = caso.input.split()
        a, b = int(a), int(b)
        if op == "+":
            return a + b
        return None
    except Exception:
        return None


def handler_memory(caso: CasoBenchmark) -> Any:
    """Simula recordar"""
    if caso.input.startswith("recuerda"):
        return caso.input.replace("recuerda", "").strip()
    if caso.input.startswith("recordar"):
        clave = caso.input.replace("recordar", "").strip()
        if "no_existe" in clave:
            return None
        return f"{clave} = ???"
    return None


def handler_planning(caso: CasoBenchmark) -> Any:
    """Simula detección de intención"""
    from reasoning.planner_v3 import Planner3
    p = Planner3()
    return p.detectar_intencion(caso.input)


def handler_rl(caso: CasoBenchmark) -> Any:
    """Simula step de RL"""
    return "step_ok" if "3x3" in caso.input else "entrena_ok"


def handler_learning(caso: CasoBenchmark) -> Any:
    """Simula accept/reject"""
    if "0.1" in caso.input and "0.9" in caso.input:
        return "rechazar"
    if "mejora" in caso.input:
        return "aceptar"
    return None


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO BENCHMARK 4.0")
    print("=" * 60)
    
    # Crear benchmark
    bench = Benchmark(seed=42)
    bench.registrar_handler("math", handler_math)
    bench.registrar_handler("memory", handler_memory)
    bench.registrar_handler("planning", handler_planning)
    bench.registrar_handler("rl", handler_rl)
    bench.registrar_handler("learning", handler_learning)
    bench.cargar_casos()
    
    print(f"\n{bench}")
    print(f"Casos cargados: {len(bench.casos)}")
    
    # Distribución
    por_cat = {}
    por_split = {}
    for c in bench.casos:
        por_cat[c.categoria] = por_cat.get(c.categoria, 0) + 1
        por_split[c.split] = por_split.get(c.split, 0) + 1
    
    print(f"\n📂 Por categoría:")
    for cat, n in sorted(por_cat.items()):
        print(f"   {cat}: {n}")
    
    print(f"\n🔀 Por split:")
    for split, n in sorted(por_split.items()):
        print(f"   {split}: {n}")
    
    # Ejecutar
    print(f"\n🚀 Ejecutando benchmark...")
    t0 = time.time()
    reporte = bench.ejecutar()
    t_total = time.time() - t0
    print(f"   Tiempo total: {t_total*1000:.1f}ms")
    
    # Mostrar reporte
    print(f"\n{reporte.resumen_texto()}")
    
    # Persistencia
    print(f"\n💾 Test persistencia:")
    tmp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_benchmark_v4.json"
    )
    reporte.guardar(tmp)
    print(f"   Guardado: {tmp}")
    print(f"   Existe: {os.path.exists(tmp)}")
    os.unlink(tmp)
    
    print("\n✅ BENCHMARK 4.0 FUNCIONANDO")
