"""Small, deterministic checks for the canonical Cerebro 1.0 path."""

import numpy as np

from cerebro_zero import Cerebro
from cerebro_zero.training import LocalTrainingRunner, TrainingJob


def test_public_api_smoke():
    ai = Cerebro()
    result = ai.run("golden path")
    assert result.success
    assert isinstance(result.text, str)
    ai.close()


def test_tiny_training_reduces_or_matches_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job = TrainingJob(
        job_id="golden-path",
        model_base="cerebro-tiny",
        dataset_version="synthetic-v1",
        tokenizer="test",
        hyperparameters={
            "vocab_size": 16,
            "d_model": 8,
            "num_heads": 2,
            "d_ff": 16,
            "num_layers": 1,
            "max_len": 4,
            "lr": 1e-2,
        },
        seed=42,
    )
    np.random.seed(42)
    model, trainer, completed = LocalTrainingRunner().run_smoke(job, steps=3)
    assert completed.status == "completed"
    assert completed.metrics["steps"] == 3
    assert np.isfinite(completed.metrics["train_loss"])
    assert model.num_parameters() > 0
    assert trainer.history["train_loss"]


def test_active_package_does_not_depend_on_historical_namespaces():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "cerebro_zero"
    forbidden = ("from legacy", "import legacy", "from v5", "import v5")
    violations = []
    for path in root.rglob("*.py"):
        if path.parts[-2:] == ("compat", "v5.py"):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                violations.append(f"{path}: {marker}")
    assert not violations, "Historical dependency in active package: " + ", ".join(violations)
