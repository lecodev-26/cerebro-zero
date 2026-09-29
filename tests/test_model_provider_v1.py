import json
import os
from cerebro_zero.models.providers import OpenAICompatibleProvider, OpenRouterProvider

class FakeResponse:
    def __init__(self, payload): self.payload=payload
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def read(self): return json.dumps(self.payload).encode()

def test_openai_compatible_provider_parses_response(monkeypatch):
    import cerebro_zero.models.providers as m
    seen={}
    def fake(req, timeout):
        seen["url"]=req.full_url; seen["auth"]=req.get_header("Authorization")
        return FakeResponse({"model":"teacher-x","choices":[{"message":{"content":"respuesta"}}],"usage":{"total_tokens":3}})
    monkeypatch.setattr(m.urllib.request,"urlopen",fake)
    monkeypatch.setenv("CEREBRO_API_KEY","secret")
    r=OpenAICompatibleProvider(model="teacher-x").generate([{"role":"user","content":"hola"}])
    assert r.text=="respuesta" and r.model=="teacher-x"
    assert seen["auth"]=="Bearer secret" and seen["url"].endswith("/chat/completions")

def test_openrouter_uses_dedicated_key(monkeypatch):
    import cerebro_zero.models.providers as m
    monkeypatch.setenv("OPENROUTER_API_KEY","or-secret")
    monkeypatch.setattr(m.urllib.request,"urlopen",lambda req,timeout: FakeResponse({"choices":[{"message":{"content":"ok"}}]}))
    assert OpenRouterProvider().generate([{"role":"user","content":"x"}]).text=="ok"
