# Cerebro Zero 1.0 — Migration Map

## Purpose

This document maps the current V5 repository into the definitive 1.0 architecture. No implementation is deleted by this map; historical functionality is first classified, then migrated behind stable interfaces.

## Current inventory

The repository currently contains the verified V5 runtime plus earlier neural, Transformer, memory, planning, security, tool and training layers.

| Current area | 1.0 destination | Action |
|---|---|---|
| `v5/runtime.py` | `cerebro_zero/cognition/runtime.py` | Extract and redesign behind interfaces |
| `v5/context.py` | `cerebro_zero/cognition/context.py` | Preserve concepts, unify API |
| `v5/evaluation.py` | `cerebro_zero/evaluation/runtime.py` | Integrate into evaluator layer |
| `v5/security.py` | `cerebro_zero/security/runtime.py` | Merge with security policy |
| `v5/skills.py` | `cerebro_zero/cognition/skills.py` | Preserve, redesign API |
| `core/*` | `cerebro_zero/core/*` | Preserve mathematical foundations |
| `models/*` | `cerebro_zero/models/*` | Unify model interfaces |
| `language/*` | `cerebro_zero/models/tokenization/*` | Unify tokenizer contract |
| `memory/*` | `cerebro_zero/memory/*` | Consolidate memory implementations |
| `reasoning/*` | `cerebro_zero/cognition/planning/*` | Merge planner/reasoning APIs |
| `tools/*` | `cerebro_zero/tools/*` | Build stable registry/policy layer |
| `security/*` | `cerebro_zero/security/*` | Merge permissions/sandbox/limits |
| `training/*` | `cerebro_zero/training/*` | Split runtime learning from model training |
| `evaluation/*` | `cerebro_zero/evaluation/*` | Consolidate benchmarks/evaluators |
| `config/*` | `cerebro_zero/config/*` | Stable configuration service |
| `mobile/*` | `cerebro_zero/compute/mobile/*` | Optional optimization backend |
| `utils/*` | `cerebro_zero/observability/*` | Move reusable runtime utilities |
| `agent/*` | historical compatibility | Replace with unified runtime |
| `cli.py` | `cerebro_zero/cli/*` | New public CLI |
| root `__init__.py` | `cerebro_zero/__init__.py` | New public import surface |

## Historical treatment

`legacy/` remains archival. V2–V5 names are not the public architecture of 1.0. Existing tests become migration/regression assets until their behavior is represented by 1.0 tests.

## Critical findings

1. The current package is a flat collection of top-level modules rather than a single `cerebro_zero` Python namespace.
2. `pyproject.toml` currently exposes `cli` and historical top-level packages directly.
3. V5 runtime is tightly coupled to its local module layout.
4. The current model stack is NumPy-based and already contains Transformer/autograd foundations.
5. The repository has training, tokenizer, memory, security and tool primitives that can be reused instead of discarded.
6. The current CLI and README expose V5-specific classes; these must be replaced by the stable 1.0 facade.
7. Existing `cerebro_zero.egg-info` is generated metadata and should not define architecture.
8. Existing artifacts/logs/database-like files need an explicit runtime-data policy before packaging.

## Migration rules

- Do not perform a destructive rewrite in one commit.
- Introduce the new namespace alongside the verified V5 implementation.
- Add compatibility/adapters while migrating behavior.
- Move tests from implementation-specific imports to public 1.0 APIs progressively.
- Keep V5 benchmark reproducibility available until equivalent 1.0 benchmarks pass.
- Delete or archive old public entry points only after clean-install tests prove the new package works.

## First implementation slice

The first coding slice should create only the package foundation:

```text
cerebro_zero/
  __init__.py
  api.py
  config/
  core/
  models/
  cognition/
  memory/
  tools/
  security/
  evaluation/
```

Then introduce a minimal stable `Cerebro` facade and provider interfaces without yet replacing the working V5 internals.

## Next migration sequence

1. Package namespace.
2. Stable `Cerebro` facade.
3. Model provider interface.
4. Adapter around current Transformer/NumPy model.
5. Cognitive runtime adapter around V5.
6. Memory service adapter.
7. Tool registry adapter.
8. Security/policy adapter.
9. Configuration service.
10. CLI adapter.
11. Regression tests through public API.
12. Packaging and clean-install qualification.

Only after these foundations are green should we begin PyTorch training infrastructure and external LLM providers.
