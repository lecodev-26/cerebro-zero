from dataclasses import dataclass
import os
from pathlib import Path

def load_secret_env():
    path = Path.home() / ".config" / "cerebro-zero" / "env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value

load_secret_env()

@dataclass(frozen=True)
class Settings:
    profile: str = "local"
    data_dir: str = "~/.cerebro-zero"
    model: str = "cerebro-local"
    max_actions: int = 32
    max_risk: float = 1.0
    token_budget: int = 4096
    tool_budget: int = 16
    memory_budget: int = 8
    provider: str = "local"
    teacher_model: str = "openrouter/free"
    provider_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(profile=os.getenv("CEREBRO_PROFILE", "local"),
                   data_dir=os.getenv("CEREBRO_DATA_DIR", "~/.cerebro-zero"),
                   model=os.getenv("CEREBRO_MODEL", "cerebro-local"),
                   provider=os.getenv("CEREBRO_PROVIDER", "local"),
                   teacher_model=os.getenv("CEREBRO_TEACHER_MODEL", "openrouter/free"),
                   provider_base_url=os.getenv("CEREBRO_PROVIDER_BASE_URL", "https://openrouter.ai/api/v1"),
                   max_actions=int(os.getenv("CEREBRO_MAX_ACTIONS", "32")),
                   token_budget=int(os.getenv("CEREBRO_TOKEN_BUDGET", "4096")))
