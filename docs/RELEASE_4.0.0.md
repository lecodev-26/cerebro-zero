# Release 4.0.0 — Scientific Validation Record

**Date:** 2026-09-28  
**Platform:** Samsung A16 / Termux / Android / aarch64  
**Python:** 3.14.6  
**NumPy:** 2.4.4

## Release gate

The release gate consists of:

```bash
python -m pytest -q
python cli.py benchmark
python -m build
cerebro-zero --version
```

### Recorded validation

- **Tests collected:** 762
- **Tests passed:** 762
- **E2E cases:** 6
- **E2E correct:** 6
- **E2E errors:** 0
- **Package:** sdist + wheel built successfully
- **CLI:** `Cerebro Zero 4.0.0`

## E2E benchmark

The benchmark is implemented in `evaluation/benchmark_e2e.py`.

It creates a real `CerebroV3` instance and exercises:

1. Arithmetic tool execution.
2. Second arithmetic operation.
3. Learning into world memory.
4. Stateful memory recall.
5. Planner execution.
6. Unknown-input handling.

The release run returned 100% case accuracy. This is a deterministic release-validation scenario set, not a claim that the system has general intelligence or universal benchmark performance.

## Integrity checks

- Active package contains no pickle-based model artifacts.
- Historical pickle files remain under `legacy/models/`.
- CI uses a single workflow with Python 3.10–3.14.
- Packaging is validated with `python -m build`.
- The README badge points to the unified CI workflow.

## Reproduction

Clone the repository, install development dependencies, then run the release gate. Record the resulting Git SHA and environment versions alongside benchmark output.

## Scope note

The project remains a from-scratch educational research system. The release is an engineering milestone: reproducible tests, honest benchmarks, packaging, documentation and explicit security boundaries.
