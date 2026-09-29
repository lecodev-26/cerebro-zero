"""Continual-growth training for the single Cerebro model lineage.

Each accepted continuation starts from the latest checkpoint, preserves
compatible weights, increases model capacity, and records the parent hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.serialization import safe_load_dict
from models.transformer import Transformer


@dataclass(frozen=True)
class GrowthStage:
    generation: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int


# Capacity grows monotonically. The model identity does not change.
GROWTH_SCHEDULE = (
    GrowthStage(0, 128, 4, 512, 4),
    GrowthStage(1, 160, 4, 640, 4),
    GrowthStage(2, 192, 4, 768, 5),
    GrowthStage(3, 224, 4, 896, 6),
    GrowthStage(4, 256, 4, 1024, 6),
)


def checkpoint_hash(path: str | Path) -> str:
    """Hash the checkpoint's JSON and NPY files deterministically."""
    base = Path(path)
    files = sorted(base.parent.glob(base.name + "*"))
    h = hashlib.sha256()
    for file in files:
        if file.is_file():
            h.update(file.name.encode())
            h.update(file.read_bytes())
    return h.hexdigest()


def load_checkpoint_state(path: str | Path) -> dict:
    return safe_load_dict(str(path))


def _copy_overlap(dst: np.ndarray, src: np.ndarray) -> None:
    """Copy the common hyper-rectangle from src into dst."""
    if dst.ndim != src.ndim:
        raise ValueError(f"Cannot expand tensor rank {src.ndim} -> {dst.ndim}")
    slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    dst[slices] = src[slices]


def expand_from_checkpoint(old_model: Transformer, state: dict, stage: GrowthStage) -> Transformer:
    """Create the next wider/deeper model and transplant old weights."""
    new_model = Transformer(
        vocab_size=old_model.vocab_size,
        d_model=stage.d_model,
        num_heads=stage.num_heads,
        d_ff=stage.d_ff,
        num_layers=stage.num_layers,
        max_len=old_model.max_len,
    )
    old_params = old_model.parameters()
    new_params = new_model.parameters()
    count = min(len(old_params), len(new_params))
    for i in range(count):
        _copy_overlap(new_params[i].data, old_params[i].data)
    return new_model


def stage_for_generation(generation: int) -> GrowthStage:
    """Return a deterministic capacity stage; extrapolate after the schedule."""
    if generation < len(GROWTH_SCHEDULE):
        return GROWTH_SCHEDULE[generation]
    last = GROWTH_SCHEDULE[-1]
    step = generation - last.generation
    d = last.d_model + 32 * step
    return GrowthStage(generation, d, 4, d * 4, last.num_layers + (step + 1) // 2)


def estimate_parameters(vocab_size: int, context_length: int, stage: GrowthStage) -> int:
    """Estimate/compute the Transformer parameter count before construction."""
    d, ff, layers = stage.d_model, stage.d_ff, stage.num_layers
    return 2 * vocab_size * d + context_length * d + layers * (4 * d * d + 2 * d * ff + 6 * d) + 2 * d


def write_lineage(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
def load_model_for_continuation(checkpoint: str | Path, manifest: dict) -> tuple[Transformer, dict]:
    """Rebuild the parent model exactly as recorded by its manifest."""
    cfg = manifest["config"]
    model = Transformer(
        vocab_size=int(manifest["vocab_size"]),
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        d_ff=int(cfg["d_ff"]),
        num_layers=int(cfg["num_layers"]),
        max_len=int(cfg["context_length"]),
    )
    state = load_checkpoint_state(checkpoint)
    params = model.parameters()
    for i, param in enumerate(params):
        key = f"weight_{i}"
        if key not in state:
            raise ValueError(f"Checkpoint missing {key}")
        if tuple(state[key].shape) != tuple(param.data.shape):
            raise ValueError(f"Parent checkpoint shape mismatch at {key}")
        param.data = state[key].copy()
    return model, state


def next_generation_manifest(parent_manifest: dict, parent_hash: str, stage: GrowthStage, parameters: int) -> dict:
    """Build the immutable lineage record for the next training generation."""
    return {
        "model_id": "cerebro-model",
        "model_name": "Cerebro Model",
        "lineage": "cerebro-model",
        "generation": stage.generation,
        "parent_checkpoint_hash": parent_hash,
        "parameters": parameters,
        "capacity_stage": {
            "d_model": stage.d_model,
            "num_heads": stage.num_heads,
            "d_ff": stage.d_ff,
            "num_layers": stage.num_layers,
        },
        "previous_manifest": parent_manifest.get("generation", 0),
    }
__all__ = [
    "GrowthStage",
    "GROWTH_SCHEDULE",
    "checkpoint_hash",
    "load_checkpoint_state",
    "expand_from_checkpoint",
    "stage_for_generation",
    "estimate_parameters",
    "load_model_for_continuation",
    "next_generation_manifest",
    "write_lineage",
]
