from .config import Settings
from .cognition.runtime import CognitiveRuntime, RunResult
from .models import LocalEchoProvider

class Cerebro:
    """Stable public 1.0 facade."""
    api_version="1"
    runtime_version="1.0.0"
    def __init__(self, settings=None, provider=None, **kwargs):
        self.settings=settings or Settings.from_env()
        self.runtime=CognitiveRuntime(self.settings, provider or LocalEchoProvider())
    def run(self, prompt, *, observation=None, constraints=None):
        if not isinstance(prompt,str) or not prompt.strip(): raise ValueError("prompt must be a non-empty string")
        return self.runtime.run(prompt, observation, constraints)
    def chat(self, prompt, **kwargs): return self.run(prompt, **kwargs)
    def close(self): return self.runtime.close()
    def stats(self): return self.runtime.stats()
