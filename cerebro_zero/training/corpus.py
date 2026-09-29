import json
from pathlib import Path
import random
from .contracts import DatasetRecord

class TeacherDatasetLoader:
    """Loads the canonical JSONL dataset artifact produced by TrainingPipeline."""
    def load(self, path):
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(DatasetRecord(**json.loads(line)))
        return rows

class CorpusBuilder:
    """Turns validated instruction/response records into deterministic LM corpora."""
    def build(self, records, seed=42, train_fraction=.90, validation_fraction=.05):
        rows = list(records)
        random.Random(seed).shuffle(rows)
        n = len(rows)
        if n < 3:
            raise ValueError("At least 3 records are required for train/validation/test")
        test_n = max(1, int(n * (1.0 - train_fraction - validation_fraction)))
        val_n = max(1, int(n * validation_fraction))
        while test_n + val_n >= n:
            if val_n > 1:
                val_n -= 1
            else:
                test_n -= 1
        a = n - test_n - val_n
        b = a + val_n
        splits = {"train": rows[:a], "validation": rows[a:b], "test": rows[b:]}
        return {k: self._render(v) for k, v in splits.items()}

    def _render(self, records):
        return "\n".join(
            f"<BOS>\nUsuario: {r.instruction}\nAsistente: {r.response}\n<EOS>"
            for r in records
        ) + "\n"
