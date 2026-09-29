from pathlib import Path
import json
from .settings import Settings

class ConfigLoader:
    def __init__(self, path=None):
        self.path=Path(path or "~/.cerebro-zero/config.json").expanduser()
    def load(self):
        data={}
        if self.path.exists():
            data=json.loads(self.path.read_text(encoding="utf-8"))
        env=Settings.from_env().__dict__
        env.update({k:v for k,v in data.items() if v is not None})
        return Settings(**{k:env[k] for k in Settings.__dataclass_fields__ if k in env})
    def save(self, settings):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(settings.__dict__,indent=2),encoding="utf-8")
        return self.path
