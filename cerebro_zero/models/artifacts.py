from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib
@dataclass(frozen=True)
class ModelArtifact:
    name:str; version:str; checkpoint:str; lineage_parent:str|None=None; metadata:dict|None=None
class ModelArtifactRegistry:
    def __init__(self,root="~/.cerebro-zero/models"):
        self.root=Path(root).expanduser(); self.root.mkdir(parents=True,exist_ok=True)
    def register(self,artifact):
        p=self.root/f"{artifact.name}-{artifact.version}.json"; data=asdict(artifact); data["checkpoint_hash"]=hashlib.sha256(Path(artifact.checkpoint).read_bytes()).hexdigest() if Path(artifact.checkpoint).exists() else None; p.write_text(json.dumps(data,indent=2),encoding="utf-8"); return p
    def list(self): return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"))]
