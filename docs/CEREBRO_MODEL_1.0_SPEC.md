# Cerebro Model 1.0 — Technical Specification

This freezes the starting specification so contributors can focus on training and evaluation.

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

## Lineage

Cerebro Model 1.0 -> 1.1 -> 1.2 ...

Every accepted model records parent checkpoint, dataset fingerprint, tokenizer, training configuration and evaluation report.
