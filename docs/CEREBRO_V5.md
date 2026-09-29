# Cerebro Zero V5
## Cognitive Autonomous Runtime

V5.0.0 is the current release. It integrates observation, context, multi-layer memory, structured reasoning, hierarchical planning, guarded actions, verification, learning and consolidation in one deterministic cognitive loop.

Loop: `observe -> context -> retrieve -> reason -> plan -> guard -> act -> verify -> learn -> consolidate`.

### Components
- `v5.runtime`: cognitive cycle, planning, tools, learning and orchestration.
- `v5.context`: bounded execution context and compression.
- `v5.security`: action and risk budgets plus guard validation.
- `v5.skills`: procedural skill registration, selection and usage tracking.
- `v5.evaluation`: outcome and calibration metrics.
- `evaluation/benchmark_v5.py`: reproducible long-running benchmark.

### Release qualification
- Package version: `5.0.0`.
- Full suite: `770 passed`.
- Release benchmark: `1000/1000` successful cycles.
- Release benchmark throughput: `195.66 cycles/s` on the Termux release environment.
- Default action execution is safe/simulated unless a tool is explicitly registered.

### V4 foundation
V5 keeps the verified V4 ML/Transformer stack intact. V4 release documents remain historical validation records rather than the current release description.
