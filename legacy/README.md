# 📦 Legacy — Código histórico de Cerebro Zero

Esta carpeta contiene **código histórico** reemplazado por versiones más recientes.

**Nada de aquí se ejecuta en producción.** Se conserva por referencia.

---

## 📁 Estructura

### `brains/` — Cerebros antiguos
- `brain.py` — Brain v1 (usa pickle)
- `brain_v2.py` — Brain v2
- `cerebro_zero_v1.py` — Cerebro Zero v1
- `central.py` — Cerebro Central 2.1
- `pipeline.py` — Pipeline 2.1

**Reemplazados por:** `agent/cerebro_v3.py`

### `benchmarks/` — Benchmarks antiguos
- `benchmark.py` — Benchmark 1.0
- `benchmark_v2.py` — Benchmark 2.0 (800 casos)
- `benchmark_v3.py` — Benchmark 3.0

**Reemplazados por:** `evaluation/benchmark_v4.py`

### `registries/` — Registros de tools antiguos
- `registry.py` — Registry 1.0
- `registry_v2.py` — Registry 2.0
- `tool_system.py` — Tool System 1.0

**Reemplazados por:** `tools/tool_system_v3.py`

### `training/` — Entrenamiento antiguo
- `continuous_learning.py` — Continuous Learning v1

**Reemplazados por:** `training/continuous_learning_v2.py` y `training/learning_engine.py`

### `dashboard/` — Dashboard 2.1
Dashboard Flask completo atado a `central.py` (legacy).

**Reemplazable en el futuro** cuando se migre a CerebroV3.

### `experiments/` — 30 scripts de experimentación
Scripts antiguos (Sept 2026). No forman parte del pipeline.

### `tools/` — Herramientas antiguas
- `clima_real.py`, `noticias_real.py`
- `herramientas*.py` (4 variantes)
- `voz.py`, `voz_real.py`

Herramientas standalone sin integración con CerebroV3.

### `evaluation/` — Módulos de evaluación antiguos
- `security.py`, `optimization.py`, `self_evaluation.py`

Sin uso en el árbol activo.

---

## 🚫 Política

- ❌ **NO** importar desde `legacy/` en código activo
- ❌ **NO** añadir código nuevo a `legacy/`
- ❌ **NO** ejecutar tests de `legacy/` en CI (excluidos vía `pytest.ini`)

Si algo de `legacy/` se necesita:
1. Migrarlo a una versión actual
2. Adaptarlo a la API actual
3. Testearlo

---

## 🔄 Historia

- **4.1** (Sept 2026): Primera reorganización. Todo este código se movió aquí para limpiar el árbol activo.

---

## 📊 Números

| Categoría | Archivos | Tests movidos |
|-----------|----------|---------------|
| brains | 5 | 2 |
| benchmarks | 3 | 1 |
| registries | 3 | 2 |
| training | 1 | — |
| dashboard | 3 | 1 |
| experiments | 30 | — |
| tools | 9 | — |
| evaluation | 3 | — |
| **TOTAL** | **57** | **6 archivos, 76 tests** |
