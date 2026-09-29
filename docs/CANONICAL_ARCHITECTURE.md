# Canonical Architecture

## Rule

Cerebro Zero 1.x is the only active product architecture. The `cerebro_zero/` package is the public namespace.

```text
CLI / SDK / HTTP
       |
       v
cerebro_zero.api.Cerebro
       |
       +--> cognition/runtime
       |       +--> memory
       |       +--> planning/reasoning
       |       +--> tools
       |       +--> security
       |
       +--> models
       |       +--> registry
       |       +--> local Transformer
       |       +--> checkpoint / lineage
       |
       +--> training
       |       +--> datasets / corpus
       |       +--> teacher/data factory
       |       +--> training runner
       |
       +--> evaluation / observability / compute
```

## Dependency direction

- Public callers depend on `cerebro_zero`.
- Runtime services may depend on `cerebro_zero.core`, model, memory, tools and security contracts.
- Models own model behavior; agents/runtime own orchestration.
- Training owns datasets, trainers and lineage; it must not become the runtime.
- Tools are invoked through the security boundary, never directly from model code.
- Evaluation observes the system; it does not become a runtime dependency.

## Historical boundary

```text
ACTIVE  ---> cerebro_zero/*

REFERENCE ONLY
  legacy/*
  v5/*
  experiments/*
  old root-level model/agent scripts

Historical code may be imported by migration tests, but active 1.x product code must not add new dependencies on it.
```

## Canonical workflow

```text
install
  -> tests
  -> tiny/reproducible training
  -> checkpoint
  -> reload
  -> evaluation
  -> lineage metadata
  -> promotion/release gate
```

The model is not the agent. The model produces model outputs; the cognitive runtime combines model, memory, planning, tools and security.

## Migration policy

1. New product code goes under `cerebro_zero/`.
2. Keep compatibility shims small and documented.
3. Do not duplicate a 1.x public service in a historical namespace.
4. Move behavior first, delete duplicate APIs only after regression coverage exists.
5. Every release must keep the canonical workflow executable from a clean install.
