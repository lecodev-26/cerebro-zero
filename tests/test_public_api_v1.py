from cerebro_zero import Cerebro, RunResult

def test_public_import_and_run():
    r=Cerebro().run("hola Cerebro")
    assert isinstance(r, RunResult)
    assert r.text == "hola Cerebro"
    assert r.execution_id and r.cycle_id

def test_cognitive_stages_are_traced():
    r=Cerebro().run("analiza esto")
    names=[e["name"] for e in r.events]
    assert names[:7] == ["request.started","observe","understand","retrieve","reason","plan","guard"]
    assert "act" in names and "verify" in names and "learn" in names and "consolidate" in names

def test_invalid_prompt_rejected():
    import pytest
    with pytest.raises(ValueError): Cerebro().run("")


def test_provider_selection_from_env(monkeypatch):
    monkeypatch.setenv("CEREBRO_PROVIDER", "openrouter")
    monkeypatch.setenv("CEREBRO_TEACHER_MODEL", "teacher-test")
    from cerebro_zero import Cerebro
    ai=Cerebro()
    assert ai.runtime.provider.name == "openrouter"
    assert ai.runtime.provider.model == "teacher-test"
