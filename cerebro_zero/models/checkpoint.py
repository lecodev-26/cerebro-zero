from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class CheckpointRef:
    path: str
    model: str
    version: str
    parent: str | None = None

class CheckpointStore:
    def __init__(self, root="~/.cerebro-zero/checkpoints"):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, ref: CheckpointRef):
        path = self.root / f"{ref.model}-{ref.version}.json"
        path.write_text(json.dumps(ref.__dict__, indent=2), encoding="utf-8")
        return path
