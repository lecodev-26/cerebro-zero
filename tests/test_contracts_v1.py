from cerebro_zero.config import Settings
from cerebro_zero.models import LocalEchoProvider
from cerebro_zero.memory import MemoryService
from cerebro_zero.tools import ToolService

def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("CEREBRO_PROFILE","test")
    assert Settings.from_env().profile == "test"

def test_memory_adapter():
    m=MemoryService(); m.add("transformer memory")
    assert m.retrieve("transformer")

def test_tool_boundary():
    t=ToolService(); t.register("add",lambda a,b:a+b)
    assert t.execute("add",{"a":2,"b":3}) == 5

def test_model_provider_contract():
    r=LocalEchoProvider().generate([{"role":"user","content":"x"}])
    assert r.text == "x"
