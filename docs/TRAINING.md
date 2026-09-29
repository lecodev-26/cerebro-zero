# Training the single Cerebro model

The repository is now organized around **one model lineage**: `cerebro-model`.
Training does not create Cerebro 1.1, 1.2, etc. Each continuation is the next
**generation of the same brain**, with a parent checkpoint and larger capacity.

## The only loop

1. Generate/refresh teacher data.
2. Train generation 0.
3. Say/run **continue**.
4. The trainer loads the latest checkpoint, preserves compatible weights,
   increases capacity, trains again and records the new parent hash.
5. Repeat.

Initial training:

    python scripts/train_cerebro_1_0.py --dataset <dataset.jsonl>

Continue the same brain:

    python scripts/train_cerebro_1_0.py --continue --dataset <dataset.jsonl>

If the dataset path is already stored in the latest manifest, `--dataset` may
be omitted. The latest lineage pointer is `artifacts/cerebro-model-1.0/LATEST.json`.

## Capacity growth

The first configured stages are deliberately reproducible, not arbitrary:

| Generation | d_model | d_ff | layers | Approx. parameters* |
|---:|---:|---:|---:|---:|
| 0 | 128 | 512 | 4 | 0.94M |
| 1 | 160 | 640 | 4 | 1.42M |
| 2 | 192 | 768 | 5 | 2.44M |
| 3 | 224 | 896 | 6 | 3.88M |
| 4 | 256 | 1024 | 6 | 5.02M |

*Approximation for a 512-token vocabulary and 128-token context; the actual
count is written to every manifest.
## What “same brain” means

A continuation starts from the previous checkpoint. Compatible tensors are
copied into the larger architecture; newly created capacity is initialized and
then trained. The result records `parent_checkpoint_hash`, generation and
parameter count, so the lineage is auditable.

**Parameters do not add arithmetically.** A 100k-parameter checkpoint followed
by another 100k-parameter continuation does not become a 200k-parameter model.
It remains a single successor model whose weights have been trained further.
If two people train independently from the same parent, they create branches;
combining them requires an explicit merge/distillation procedure. We do not
pretend that concatenating parameter files creates one larger brain.

## Shared training by contributors

Contributors can train the same lineage sequentially by starting from the
latest published checkpoint, running the continuation command, and reporting
the resulting manifest/checkpoint hash. The canonical lineage is identified as
`cerebro-model`, not by the generation number. Generation numbers only record
how many accepted capacity-growth steps have occurred.

Do not commit credentials, private datasets or local checkpoints containing
sensitive data. Teacher output must retain provenance and the applicable model
/provider terms.

## Serious-run report

Record hardware/OS, Python/NumPy versions, dataset fingerprint and record
count, tokenizer, architecture, parameter count, epochs, batch size, learning
rate, seed, train/validation loss and perplexity, held-out benchmark,
checkpoint hash and training time.

A checkpoint is a candidate until it passes the evaluation/release gate.
## Teacher data

Configure a teacher locally; never commit the key:

    export OPENROUTER_API_KEY="..."
    export CEREBRO_PROVIDER=openrouter
    export CEREBRO_TEACHER_MODEL=openrouter/free

Generate canonical data with the public API:

    from cerebro_zero import Cerebro
    ai = Cerebro()
    prompts = ai.teacher_prompts(limit=100)
    result = ai.generate_teacher_data([p.prompt for p in prompts])
    print(result["artifact_path"])

Free-provider availability is not guaranteed. Verify the selected provider and
model terms before using generated data for training.

## Smoke test

Use a small run before spending compute:

    python scripts/train_cerebro_1_0.py --dataset <dataset.jsonl> --epochs 1 --batches-per-epoch 2

The smoke run is only a plumbing check. It is not the final Cerebro Model.
