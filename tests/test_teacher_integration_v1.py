from cerebro_zero import Cerebro
from cerebro_zero.models.base import BaseModelProvider
from cerebro_zero.core.contracts import ModelResponse

class FakeProvider(BaseModelProvider):
    name = "fake"

    def generate(self, messages, **kwargs):
        return ModelResponse(
            text="This is a sufficiently detailed generated answer for the quality gate.",
            model="fake-model",
            metadata={"usage": {"total_tokens": 2}},
        )

def test_public_teacher_data_pipeline():
    ai = Cerebro(provider=FakeProvider())
    ai.teacher.providers["openrouter"] = ai.runtime.provider
    out = ai.generate_teacher_data(["Explain recursion", "Explain a loop"])
    assert out["count"] == 2
    assert len(out["train"]) == 1
    assert len(out["validation"]) == 1
    assert out["train"][0].response.startswith("This is a sufficiently")
    assert "artifact_path" in out and out["artifact_path"].endswith(".jsonl")
    assert len(out["dataset_fingerprint"]) == 16
