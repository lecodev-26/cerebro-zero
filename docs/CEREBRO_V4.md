# Cerebro Zero 4.0

## Release scope

Cerebro Zero 4.0 is the integrity-focused release built on the 3.x cognitive pipeline.

The release closes the documented 4.x engineering gaps:

- **4.4 E2E Benchmark** — measures the real `CerebroV3` pipeline.
- **4.5 E2E Testing** — exercises CLI and system-level flows.
- **4.6 Memory Indexing** — adds lexical pre-indexing and vectorized candidate scoring.
- **4.7 Transformer KV-cache** — adds incremental inference caching.
- **4.8 Security Hardening** — strengthens subprocess limits and child-process isolation.
- **4.9 Unified CI** — one release workflow validates Python 3.10–3.14, E2E, lint and packaging.
- **4.10 Scientific Release** — reproducibility and release validation are documented.
- **4.11 Documentation** — README, roadmap and architecture documentation reflect the actual code.

## Validation

Run the complete local release gate:

```bash
python -m pytest -q
python cli.py benchmark
python -m build
cerebro-zero --version
```

The Termux release validation on 28 September 2026 collected **762 tests**.

The E2E benchmark contains six integrated scenarios and, on the release validation run, produced 6/6 correct results with zero execution errors. These measurements are a local release-validation result, not a universal performance claim.

## Architecture

```text
CerebroV3
  ├── tokenizer
  ├── working / procedural / world memory
  ├── Planner 3.0
  ├── Tool System 3.0
  ├── Learning Engine 4.3
  └── reproducibility / evaluation

Evaluation
  ├── benchmark_v4.py       component benchmark
  └── benchmark_e2e.py      integrated CerebroV3 benchmark

Security
  ├── permissions
  ├── subprocess isolation
  ├── timeout / output limits
  └── platform-aware resource controls
```

## Reproducibility

Use a fixed benchmark seed (`42`) for the release benchmark.

The project records environment information through the reproducibility module, including Python, NumPy, platform and Git information.

For a release candidate, record:

1. Git commit SHA.
2. Python version.
3. NumPy version.
4. Test count and result.
5. E2E benchmark summary.
6. Package build result.

## Security boundary

The sandbox is intentionally described conservatively. It is a subprocess sandbox with resource controls and a reduced child environment. On Termux/Android it is **not** equivalent to a kernel-enforced container, namespace sandbox or seccomp policy.

## Legacy policy

Historical code and old pickle artifacts live under `legacy/` and are excluded from the active package/test path. New active code must use the JSON/NPY serialization layer.

## Release principle

4.0 is complete when the repository says what the code actually does, the tests verify the active system, the package builds, and CI validates the supported Python matrix.
