#!/usr/bin/env python3
"""Train Cerebro Model 1.0 from a canonical teacher JSONL dataset."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from cerebro_zero.training.model_config import CerebroModelConfig
from cerebro_zero.training.corpus import TeacherDatasetLoader, CorpusBuilder
from language.tokenizer_v3 import BPETokenizer
from language.dataset import TextDataset
from models.transformer import Transformer
from training.lm_trainer import LMTrainer

def main():
    p = argparse.ArgumentParser(description="Train Cerebro Model 1.0")
    p.add_argument("--dataset", required=True, help="canonical teacher JSONL")
    p.add_argument("--output", default="artifacts/cerebro-model-1.0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    args = p.parse_args()

    cfg = CerebroModelConfig(
        seed=args.seed,
        epochs=args.epochs or 3,
        batches_per_epoch=args.batches_per_epoch or 200,
        context_length=args.context_length or 128,
        d_model=args.d_model or 128,
        num_heads=args.num_heads or 4,
        d_ff=args.d_ff or 512,
        num_layers=args.num_layers or 4,
    ).validate()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    records = TeacherDatasetLoader().load(args.dataset)
    if not records:
        raise SystemExit("Dataset is empty")

    corpus = CorpusBuilder().build(records, cfg.seed, cfg.train_fraction, cfg.validation_fraction)
    for split, text in corpus.items():
        (out / f"{split}.txt").write_text(text, encoding="utf-8")

    full = "\n".join(corpus.values())
    tok = BPETokenizer(vocab_size=cfg.vocab_size)
    tok.entrenar(full)
    tok.guardar(str(out / "tokenizer.json"))

    train_ds = TextDataset(str(out / "train.txt"), tok, context_len=cfg.context_length, add_bos_eos=False)
    val_ds = TextDataset(str(out / "validation.txt"), tok, context_len=cfg.context_length, add_bos_eos=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Dataset is too small for the configured context length")

    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        max_len=cfg.context_length,
    )
    trainer = LMTrainer(model, learning_rate=cfg.learning_rate)
    history = trainer.train(
        train_ds, val_ds,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        batches_per_epoch=cfg.batches_per_epoch,
        verbose=True,
        early_stopping_patience=2,
    )
    trainer.save(str(out / "checkpoint"))

    manifest = {
        "model": "Cerebro Model 1.0",
        "config": cfg.to_dict(),
        "vocab_size": tok.vocab_size,
        "parameters": model.num_parameters(),
        "dataset": str(Path(args.dataset).resolve()),
        "history": history,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": "completed",
        "output": str(out),
        "parameters": model.num_parameters(),
        "vocab_size": tok.vocab_size,
    }, indent=2))

if __name__ == "__main__":
    main()
