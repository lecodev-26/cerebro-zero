import numpy as np
from core.tensor import Tensor
from models.transformer import Transformer
from ..core.contracts import ModelResponse
from .base import BaseModelProvider

class LocalTransformerProvider(BaseModelProvider):
    name = "cerebro-transformer"
    def __init__(self, model: Transformer):
        self.model = model
        self._cache = None
        self._position = 0

    def reset_cache(self):
        self._cache, self._position = None, 0

    def generate_ids(self, token_ids, max_new_tokens=16, temperature=1.0, top_k=None):
        ids = list(token_ids)
        self.reset_cache()
        for token in ids:
            logits, self._cache = self.model.forward_cached([token], self._cache, self._position)
            self._position += 1
        for _ in range(max_new_tokens):
            values = logits.data[0, -1] / max(temperature, 1e-6)
            if top_k:
                idx = np.argsort(values)[-top_k:]
                mask = np.full_like(values, -1e9); mask[idx] = values[idx]; values = mask
            probs = np.exp(values - values.max()); probs /= probs.sum()
            nxt = int(np.random.choice(self.model.vocab_size, p=probs))
            ids.append(nxt)
            logits, self._cache = self.model.forward_cached([nxt], self._cache, self._position)
            self._position += 1
        return ids

    def generate(self, messages, **kwargs):
        raise NotImplementedError("Tokenizer/model vocabulary adapter required")

    def capabilities(self):
        return {"generation": True, "training": True, "kv_cache": True}
