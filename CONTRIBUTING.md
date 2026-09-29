# Contributing to Cerebro Zero

The project is now in the model-training phase. The most valuable contributions are reproducible training runs, data-quality improvements, benchmark improvements and training-performance improvements.

## Before changing code

Run:

    pytest -q

Read:

- docs/CEREBRO_MODEL_1.0_SPEC.md
- docs/TRAINING.md
- docs/DATASET_POLICY.md

## Training runs

Use the Model training issue template. Report the full configuration, dataset fingerprint, metrics and checkpoint hash. Never publish credentials or private data.

A training result is a candidate until it passes the evaluation gate.

## Dataset contributions

Every record needs provenance and terms/license information. Do not add held-out benchmark data to training data.

## Code contributions

1. Fork the repository.
2. Create a branch.
3. Make one focused change.
4. Run the full test suite.
5. Include reproducibility information when the change affects training.
6. Open a pull request.

## Scope

Preferred work:

- model architecture experiments
- tokenizer improvements
- teacher/data factory improvements
- dataset quality and deduplication
- evaluation and contamination checks
- CPU/GPU training backends
- checkpoint loading and promotion
- reproducibility
- documentation

Avoid unrelated rewrites while the 1.0 model is being trained.

## Security

Never commit API keys, tokens, credentials, private datasets or generated secret files.
