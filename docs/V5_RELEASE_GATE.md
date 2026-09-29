# Cerebro Zero V5 — Release Gate

V5 is the autonomous cognitive-runtime layer above the V4 ML stack. The release gate validates the runtime loop, memory, reasoning, planning/repair, tools/permissions, context, action/risk budgets, skills, evaluation, learning, replay, reproducible benchmark, and Android/Termux execution.

## Runtime loop

`observe -> context -> retrieve -> reason -> plan -> guard -> act -> verify -> learn -> consolidate`

The V5 runtime is deterministic at the orchestration level and uses a safe simulated executor by default. Real external tools must be explicitly registered with permissions and risk metadata.

## Qualification commands

- `python -m pytest -q`
- `python -m evaluation.benchmark_v5 --cycles 100`
- `python -m evaluation.benchmark_v5 --cycles 1000`

The benchmark reports success rate, elapsed time, cycles/second, and runtime stats. Results are environment-specific and are not treated as hardware-independent performance claims.

## Transformer V5 scope

V5 preserves and exercises the existing transformer/ML stack from V4. V5 does not replace the model architecture with an unvalidated experimental transformer; model experiments remain isolated from the autonomous runtime so the cognitive loop stays reproducible.

## Android/Termux qualification

The authoritative local qualification target is the Android/Termux environment used for this repository. The full test suite and long-running benchmark are executed there before release. No broader device-performance claim is made from a single handset.
