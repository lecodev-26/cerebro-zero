#!/usr/bin/env python3
"""Train or continue the single Cerebro Model lineage.

First run creates generation 0. Every --continue run loads the latest
checkpoint, preserves compatible weights, increases capacity, and trains
that successor as the same Cerebro model lineage.
"""
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
from cerebro_zero.training.continual import (
    checkpoint_hash, expand_from_checkpoint, load_model_for_continuation,
    stage_for_generation, estimate_parameters, next_generation_manifest,
)
from language.tokenizer_v3 import BPETokenizer
from language.dataset import TextDataset
from models.transformer import Transformer
from training.lm_trainer import LMTrainer


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def train_model(model, tok, out, cfg, corpus):
    for split, text in corpus.items():
        (out / f"{split}.txt").write_text(text, encoding="utf-8")
    train_ds = TextDataset(str(out / "train.txt"), tok, context_len=cfg.context_length, add_bos_eos=False)
    val_ds = TextDataset(str(out / "validation.txt"), tok, context_len=cfg.context_length, add_bos_eos=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Dataset is too small for the configured context length")
    trainer = LMTrainer(model, learning_rate=cfg.learning_rate)
    history = trainer.train(
        train_ds, val_ds, epochs=cfg.epochs, batch_size=cfg.batch_size,
        batches_per_epoch=cfg.batches_per_epoch, verbose=True,
        early_stopping_patience=2,
    )
    trainer.save(str(out / "checkpoint"))
    return history


def main():
    p = argparse.ArgumentParser(description="Train/continue the single Cerebro Model")
    p.add_argument("--dataset", help="canonical teacher JSONL; required for first run")
    p.add_argument("--output", default="artifacts/cerebro-model-1.0")
    p.add_argument("--continue", dest="continuation", action="store_true",
                   help="continue from the latest checkpoint and grow capacity")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batches-per-epoch", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.output)
    latest_path = root / "LATEST.json"
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.continuation:
        if not latest_path.exists():
            raise SystemExit("No LATEST.json found. Run the initial training first.")
        latest = load_json(latest_path)
        parent_dir = root / latest["generation_dir"]
        parent_manifest = load_json(parent_dir / "manifest.json")
        dataset_path = Path(args.dataset or parent_manifest["dataset"])
        if not dataset_path.exists():
            raise SystemExit(f"Dataset not found: {dataset_path}")
        parent_checkpoint = parent_dir / "checkpoint"
        parent_model, _ = load_model_for_continuation(parent_checkpoint, parent_manifest)
        generation = int(parent_manifest["generation"]) + 1
        stage = stage_for_generation(generation)
        estimated = estimate_parameters(parent_model.vocab_size, parent_model.max_len, stage)
        print(f"🧠 Continuing the SAME Cerebro model: generation {generation}")
        print(f"   Parent: generation {parent_manifest['generation']} | parameters={parent_manifest['parameters']}")
        print(f"   Next capacity: d={stage.d_model}, ff={stage.d_ff}, layers={stage.num_layers}")
        print(f"   Estimated parameters: {estimated}")
        model = expand_from_checkpoint(parent_model, {}, stage)
        tokenizer_path = parent_dir / "tokenizer.json"
        tok = BPETokenizer.cargar(str(tokenizer_path))
        corpus = CorpusBuilder().build(TeacherDatasetLoader().load(dataset_path), args.seed)
        cfg = CerebroModelConfig(
            epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
            seed=args.seed, context_length=parent_model.max_len,
            d_model=stage.d_model, num_heads=stage.num_heads,
            d_ff=stage.d_ff, num_layers=stage.num_layers,
        ).validate()
        out = root / f"generation-{generation:03d}"
        out.mkdir(parents=True, exist_ok=True)
        parent_hash = checkpoint_hash(parent_checkpoint)
        history = train_model(model, tok, out, cfg, corpus)
        manifest = next_generation_manifest(parent_manifest, parent_hash, stage, model.num_parameters())
        manifest.update({
            "config": cfg.to_dict(), "vocab_size": tok.vocab_size,
            "dataset": str(dataset_path.resolve()), "history": history,
            "tokenizer": "tokenizer.json", "checkpoint": "checkpoint",
            "status": "candidate",
        })
        save_json(out / "manifest.json", manifest)
        save_json(latest_path, {"lineage": "cerebro-model", "generation": generation,
                                "generation_dir": out.name, "checkpoint_hash": checkpoint_hash(out / "checkpoint")})
        print(json.dumps({"status":"completed", "lineage":"cerebro-model",
                          "generation":generation, "parameters":model.num_parameters(),
                          "parent_checkpoint_hash":parent_hash}, indent=2))
        return

    if not args.dataset:
        raise SystemExit("--dataset is required for the first training run")
    records = TeacherDatasetLoader().load(args.dataset)
    if not records:
        raise SystemExit("Dataset is empty")
    cfg = CerebroModelConfig(epochs=args.epochs, batches_per_epoch=args.batches_per_epoch, seed=args.seed).validate()
    corpus = CorpusBuilder().build(records, cfg.seed, cfg.train_fraction, cfg.validation_fraction)
    out = root / "generation-000"
    out.mkdir(parents=True, exist_ok=True)
    full = "\n".join(corpus.values())
    tok = BPETokenizer(vocab_size=cfg.vocab_size)
    tok.entrenar(full)
    tok.guardar(str(out / "tokenizer.json"))
    model = Transformer(vocab_size=tok.vocab_size, d_model=cfg.d_model,
                        num_heads=cfg.num_heads, d_ff=cfg.d_ff,
                        num_layers=cfg.num_layers, max_len=cfg.context_length)
    history = train_model(model, tok, out, cfg, corpus)
    manifest = {"model_id":"cerebro-model", "model_name":"Cerebro Model",
                "lineage":"cerebro-model", "generation":0,
                "parameters":model.num_parameters(), "vocab_size":tok.vocab_size,
                "config":cfg.to_dict(), "dataset":str(Path(args.dataset).resolve()),
                "history":history, "tokenizer":"tokenizer.json", "checkpoint":"checkpoint",
                "status":"candidate"}
    save_json(out / "manifest.json", manifest)
    save_json(latest_path, {"lineage":"cerebro-model", "generation":0,
                            "generation_dir":out.name, "checkpoint_hash":checkpoint_hash(out / "checkpoint")})
    print(json.dumps({"status":"completed", "lineage":"cerebro-model", "generation":0,
                      "parameters":model.num_parameters(), "next":"python scripts/train_cerebro_1_0.py --continue"}, indent=2))


if __name__ == "__main__":
    main()
