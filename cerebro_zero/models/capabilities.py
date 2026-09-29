from dataclasses import dataclass

@dataclass(frozen=True)
class ModelCapabilities:
    generation: bool = True
    training: bool = False
    tool_use: bool = False
    streaming: bool = False
    kv_cache: bool = False

    def as_dict(self):
        return self.__dict__.copy()
