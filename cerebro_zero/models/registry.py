from .spec import ModelSpec

class ModelRegistry:
    def __init__(self):
        self._models: dict[str, object] = {}
        self._specs: dict[str, ModelSpec] = {}

    def register(self, model, spec: ModelSpec):
        self._models[spec.name] = model
        self._specs[spec.name] = spec

    def get(self, name):
        return self._models[name]

    def spec(self, name):
        return self._specs[name]

    def list(self):
        return list(self._specs.values())
