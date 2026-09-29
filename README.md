# Cerebro Zero

**Cerebro Zero is an open-source cognitive AI system whose next concrete milestone is its own trained language model.**

> Current state: the 1.0 runtime, memory, tools, security, API, CLI, teacher/data pipeline and reproducible training path are implemented. The remaining core milestone is training and evaluating Cerebro Model 1.0.

[![CI](https://github.com/lecodev-26/cerebro-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/lecodev-26/cerebro-zero/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The goal

Cerebro Zero is moving from V2-V5 research stages to one unified 1.x product line:

Cerebro Runtime 1.0 + Cerebro Model 1.0 -> 1.1 -> 1.2 -> ...

V2, V3, V4 and V5 are historical engineering stages. New model releases continue the same lineage.

## What is ready

- Stable Python API: from cerebro_zero import Cerebro
- CLI: cerebro
- Cognitive loop: observe -> understand -> retrieve -> reason -> plan -> guard -> act -> verify -> learn -> consolidate
- Unified memory and retrieval
- Tool registry, permissions, risk budgets and audit
- API keys and HTTP API
- Model registry, checkpoints and lineage
- OpenRouter-compatible teacher provider
- Teacher -> dataset factory with provenance, validation, quality scoring and deduplication
- Reproducible CPU training path and safe checkpoint persistence
- Release/evaluation gates
- CI and regression suite

## The remaining core step: train the model

The repository is intentionally prepared so contributors do not need to redesign the platform before experimenting with training.

### Cerebro Model 1.0 baseline

| Component | Starting value |
|---|---:|
| Architecture | decoder-only causal Transformer |
| Tokenizer | byte-level BPE |
| Vocabulary | 512 target tokens |
| Context | 128 tokens |
| Hidden size | 128 |
| Attention heads | 4 |
| FFN | 512 |
| Layers | 4 |
| Precision | float32 |
| Objective | causal LM + instruction/SFT data |
| Seed | 42 |

These are the baseline experiment, not sacred numbers. A contributor can change them, but the run must be reproducible and benchmarked.

See docs/CEREBRO_MODEL_1.0_SPEC.md.

## Train it

### Install

    git clone https://github.com/lecodev-26/cerebro-zero.git
    cd cerebro-zero
    python -m pip install -e .

### Generate teacher data

Teacher models can generate candidate supervised data. Provider/model terms must be checked before data is used for a public model release.

    export OPENROUTER_API_KEY="YOUR_KEY"
    export CEREBRO_PROVIDER=openrouter
    export CEREBRO_TEACHER_MODEL=openrouter/free

Then:

    from cerebro_zero import Cerebro
    ai = Cerebro()
    prompts = ai.teacher_prompts(limit=10)
    prepared = ai.generate_teacher_data([p.prompt for p in prompts])
    print(prepared["artifact_path"])

### Train Cerebro Model 1.0

    python scripts/train_cerebro_1_0.py --dataset ~/.cerebro-zero/training/datasets/<dataset>.jsonl

Smoke test first:

    python scripts/train_cerebro_1_0.py --dataset <dataset>.jsonl --epochs 1 --batches-per-epoch 2

The command creates tokenizer, train/validation/test corpora, checkpoint and experiment manifest under artifacts/cerebro-model-1.0/.

## Contributing training runs

If you have more compute than the reference environment, you can train a candidate without changing the runtime.

Open a GitHub Issue using the Model training report template and include:

- hardware/OS
- dataset fingerprint and record count
- tokenizer/model configuration
- seed and hyperparameters
- train/validation loss and perplexity
- held-out benchmark results
- checkpoint hash
- training time

Pull requests are welcome for improvements to the training pipeline, data factory, evaluation suite and reproducibility tooling.

Do not upload API keys, private datasets or credentials.

See docs/TRAINING.md and docs/DATASET_POLICY.md.

## Architecture

    CLI / Python SDK / HTTP API
              |
        AI Orchestrator
              |
    cognition / memory / planning / tools / security
              |
        model registry
              |
    Cerebro Model 1.x | teacher providers for training only
              |
    evaluation -> promotion gate -> lineage

## Public API

    from cerebro_zero import Cerebro

    ai = Cerebro()
    result = ai.run("Analiza este proyecto y propón mejoras")
    print(result.text)

## Development checks

    pytest -q
    python -m build --wheel
    python scripts/release_gate.py

## Historical releases

V2-V5 remain available as historical engineering stages. The 1.x line is the canonical future line.

## License

MIT — see LICENSE.

GitHub: https://github.com/lecodev-26/cerebro-zero
