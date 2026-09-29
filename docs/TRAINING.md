# Training Cerebro Model 1.0

The repository is intentionally prepared so the remaining product work is model training, evaluation and promotion.

## Fast path

    git clone https://github.com/lecodev-26/cerebro-zero.git
    cd cerebro-zero
    python -m pip install -e .

Configure a teacher locally; never commit the key:

    export OPENROUTER_API_KEY="..."
    export CEREBRO_PROVIDER=openrouter
    export CEREBRO_TEACHER_MODEL=openrouter/free

Generate canonical data:

    from cerebro_zero import Cerebro
    ai = Cerebro()
    prompts = ai.teacher_prompts(limit=10)
    prepared = ai.generate_teacher_data([p.prompt for p in prompts])
    print(prepared["artifact_path"])

Train:

    python scripts/train_cerebro_1_0.py --dataset ~/.cerebro-zero/training/datasets/<dataset>.jsonl

Smoke first:

    python scripts/train_cerebro_1_0.py --dataset <dataset>.jsonl --epochs 1 --batches-per-epoch 2

## Report every serious run

Include hardware/OS, Python/NumPy versions, dataset fingerprint and record count, tokenizer, model dimensions, parameter count, epochs, batch size, learning rate, seed, train/validation loss and perplexity, held-out benchmark, checkpoint hash and training time.

Do not upload API keys, private datasets or credentials.

## Experiment rules

Keep a fixed held-out test set. Change one major variable at a time when possible. Never judge a model from training loss alone.

## Issues

Use GitHub Issues for training runs, dataset proposals, benchmark regressions, hardware compatibility and reproducibility reports. A training result is a candidate until it passes the evaluation gate.
