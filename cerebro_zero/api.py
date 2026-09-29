from .config import Settings
from .cognition.runtime import CognitiveRuntime, RunResult
from .models import LocalEchoProvider, OpenAICompatibleProvider, OpenRouterProvider
from .tools import ToolService
from .security import SecurityPolicy,APIKeyStore
from .training import TrainingPipeline, TeacherRegistry, TeacherService, SyntheticDataFactory

class Cerebro:
    """Stable public 1.0 facade."""
    api_version="1"
    runtime_version="1.0.0"
    def __init__(self, settings=None, provider=None, **kwargs):
        self.settings=settings or Settings.from_env()
        selected = provider or self._provider_from_settings(self.settings)
        self.runtime=CognitiveRuntime(self.settings, selected)
        self.tools=ToolService(SecurityPolicy(self.settings.max_actions,self.settings.max_risk))
        self.keys=APIKeyStore(kwargs.get("key_path")) if kwargs.get("key_path") else APIKeyStore()
        self.training=TrainingPipeline()
        self.teacher_registry=TeacherRegistry.default()
        self.teacher=TeacherService(self.teacher_registry, {self.settings.provider: selected})
        self.data_factory=SyntheticDataFactory(self.teacher_registry)
    @staticmethod
    def _provider_from_settings(settings):
        if settings.provider == "openrouter":
            return OpenRouterProvider(model=settings.teacher_model)
        if settings.provider in {"openai-compatible", "remote"}:
            return OpenAICompatibleProvider(base_url=settings.provider_base_url, model=settings.teacher_model)
        return LocalEchoProvider()

    def run(self,prompt,*,observation=None,constraints=None):
        if not isinstance(prompt,str) or not prompt.strip(): raise ValueError("prompt must be a non-empty string")
        return self.runtime.run(prompt,observation,constraints)
    def chat(self,prompt,**kwargs): return self.run(prompt,**kwargs)
    def close(self): return self.runtime.close()
    def register_tool(self,spec): return self.tools.register(spec)
    def execute_tool(self,name,arguments=None,*,approved=False): return self.tools.execute(name,arguments,approved=approved)
    def tool_catalog(self): return self.tools.describe()
    def model_catalog(self): return self.runtime.models.list()
    def generate_teacher_data(self, prompts, *, teacher_name="openrouter", max_tokens=512, temperature=0.2, save=True):
        results = self.teacher.generate_batch(prompts, teacher_name=teacher_name, max_tokens=max_tokens, temperature=temperature)
        records = self.data_factory.from_teacher_results(results)
        prepared = self.training.prepare(records)
        if save:
            path, fingerprint = self.training.save_dataset(prepared)
            prepared["artifact_path"] = str(path)
            prepared["dataset_fingerprint"] = fingerprint
        return prepared
    def stats(self): return {**self.runtime.stats(),"tools":self.tools.audit.stats()}
