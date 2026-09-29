# Cerebro Zero 1.0 — Status

Cerebro Zero 1.x is the canonical active architecture. V2–V5 and `legacy/` are historical compatibility/reference code.

## Implemented

- Stable `cerebro_zero.Cerebro` public API and CLI.
- Cognitive runtime with explicit observe → understand → retrieve → reason → plan → guard → act → verify → learn → consolidate flow.
- Model registry, local Transformer provider, checkpoints and training lineage metadata.
- NumPy tensor/autograd stack, Transformer, tokenizer and language training.
- Unified memory services and retrieval.
- Tool registry, permissions, risk budgets and audit logging.
- Security policy, API-key handling and HTTP API.
- Teacher registry/provider integration and provenance-aware synthetic data preparation.
- Evaluation/release gates, reproducibility helpers and CI.

## In progress

- Qualifying the full 1.0 release gate on clean environments.
- Expanding end-to-end reproducible training evidence for Cerebro Model 1.0.
- Tightening package boundaries so active code depends on the `cerebro_zero/` namespace first.
- Replacing historical top-level imports in active training/model code as migration permits.

## Planned

- Public, versioned Cerebro Model 1.0 training artifacts.
- Larger reproducible datasets with explicit provenance and licensing records.
- Stronger numerical gradient coverage for attention and LayerNorm.
- Fixed held-out benchmarks and regression thresholds for model promotion.
- Memory and tool benchmarks that measure their effect rather than assuming improvement.

## Experimental

- Teacher-generated training data.
- Continual lineage growth and capacity expansion.
- Local Transformer generation and KV-cache paths.

## Legacy

The following areas are retained for history or migration reference and must not receive new product dependencies:

- `legacy/`
- `v5/`
- `experiments/`
- historical root-level scripts and old model generations.

New 1.x code belongs under `cerebro_zero/` unless it is explicitly a dataset, test, migration utility or historical artifact.

## Release definition

A 1.0 release candidate is not considered complete because the modules merely exist. It must have reproducible tests, package installation, public API smoke coverage, a documented model/data lineage and a measured training/evaluation run.
