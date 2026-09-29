# Cerebro Zero 1.0 Release Gate

Cerebro Zero 1.0 is released only when the unified `cerebro_zero` public namespace, cognitive runtime, memory, tools, security, distribution, REST API, API-key scopes, owned model artifact/checkpoint, held-out benchmark, reproducible CPU training, historical V2–V5 migration and release documentation all pass their gates.

## Model lineage

`Cerebro Model 1.0 -> 1.1 -> 1.2 ...` is the intended lineage. Each model release records its parent, dataset version, tokenizer, training configuration, evaluation report and checkpoint hash.

Teacher models are training inputs, not Cerebro model identities. Teacher outputs require provenance and license/terms review before entering a training dataset.

## Compute policy

CPU/Termux is supported for development and smoke training. GPU is optional and must be selected by an explicit backend; the runtime never assumes GPU availability.
