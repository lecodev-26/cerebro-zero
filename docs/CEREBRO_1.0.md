# Cerebro Zero 1.0

Cerebro Zero 1.0 unifies the V2-V5 work under one public package and stable facade.

## Current milestone

The platform is training-ready. The remaining core milestone is the first Cerebro-owned language model checkpoint.

- Runtime: ready
- Memory: ready
- Tools/security: ready
- API/CLI/HTTP: ready
- Teacher providers: ready
- Dataset factory: ready
- Reproducible training path: ready
- Model 1.0 training: next
- Model benchmark/promotion: follows training

## Public API

    from cerebro_zero import Cerebro
    ai = Cerebro()
    response = ai.run("Analiza este proyecto y propón mejoras")
    print(response.text)

## Training

See:

- CEREBRO_MODEL_1.0_SPEC.md
- TRAINING.md
- DATASET_POLICY.md

The baseline training command is:

    python scripts/train_cerebro_1_0.py --dataset ~/.cerebro-zero/training/datasets/<dataset>.jsonl

Teacher models create candidate data. The accepted release artifact remains a Cerebro-owned checkpoint with explicit lineage.

## Model lineage

Cerebro Model 1.0 -> 1.1 -> 1.2 ...

Each accepted model records parent checkpoint, dataset fingerprint, tokenizer, training configuration and evaluation report.

V2, V3, V4 and V5 remain historical engineering stages.
