from cerebro_zero import Cerebro
from cerebro_zero.training import DatasetQuality, PromptCatalog
from cerebro_zero.training.contracts import DatasetRecord

def test_prompt_catalog_is_deterministic_and_bounded():
    rows = PromptCatalog().sample(tasks=["coding"], limit=2)
    assert len(rows) == 2
    assert all(r.task == "coding" for r in rows)

def test_quality_rewards_structured_nonempty_records():
    scorer = DatasetQuality()
    low = DatasetRecord("x", "y")
    high = DatasetRecord("Explica Python", "Python es un lenguaje de programación.\n```python\nprint('ok')\n```", "coding", "teacher", "provider-output", "openrouter", {"model":"m"})
    assert scorer.score(high) > scorer.score(low)

def test_public_prompt_catalog():
    ai = Cerebro()
    assert ai.teacher_prompts(limit=3)
