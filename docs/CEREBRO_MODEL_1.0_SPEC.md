# Cerebro Model 1.0 — Technical Specification

This freezes the starting specification so contributors can focus on training and evaluation. The repository uses one cerebro-model lineage; capacity-growth generations are not separate product models.

## Goal

Produce the first Cerebro-owned language model checkpoint. Teacher models are data sources only; they are never shipped as Cerebro Model 1.0.

## Architecture

- decoder-only causal Transformer
- byte-level BPE tokenizer
- vocabulary target: 512 tokens
- context: 128 tokens
- hidden size: 128
- attention heads: 4
- FFN: 512
- layers: 4
- float32
- causal next-token prediction followed by instruction/SFT data

This is the starting experiment, not a claim that these dimensions are optimal. Contributors may change them with reproducible benchmark evidence.

## Dataset

Canonical JSONL fields: instruction, response, task, source, license, provenance and metadata.

Required: provenance, terms/license, deterministic fingerprint, exact train/validation/test split, deduplication, quality filtering and a held-out evaluation set that is never trained on.

Target mixture: coding, reasoning, math, science, planning, analysis, instruction following, tool-use traces and general Spanish/English language ability.

## Teacher distillation

Teachers generate candidates. The factory validates, scores, deduplicates and records provenance. Free-provider availability is not assumed to be permanent.

## Training command

    python scripts/train_cerebro_1_0.py --dataset ~/.cerebro-zero/training/datasets/<dataset>.jsonl

The command writes tokenizer, split corpora, checkpoint and manifest under artifacts/.

## Acceptance gate

A checkpoint is a candidate only if training is reproducible, validation improves against baseline, the held-out benchmark passes, contamination checks pass, and checkpoint/dataset fingerprints are recorded.

## Single-lineage growth

The model identity is cerebro-model. Training generations are capacity-growth steps of that same model, not new product versions. Each continuation loads the latest checkpoint, preserves compatible weights, adds capacity according to the growth schedule, and continues optimization.

The canonical trainer records generation, parent checkpoint hash, architecture, parameter count, dataset fingerprint and evaluation history. Parameters do not add arithmetically between generations or between independently trained branches.

## Lineage

Cerebro Model 1.0 is the product/release identity. Its internal training generations are implementation checkpoints inside that identity. Future product model evolution remains Cerebro Model 1.1 -> 1.2 ... only when a genuinely new model release is declared.

Every accepted checkpoint records parent checkpoint, dataset fingerprint, tokenizer, training configuration and evaluation report.
