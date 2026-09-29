from cerebro_zero.core.contracts import ModelResponse
from cerebro_zero.models.base import BaseModelProvider
from cerebro_zero.training.teachers import TeacherRegistry, TeacherService

class FakeProvider(BaseModelProvider):
    name = "fake"
    def generate(self, messages, **kwargs):
        assert isinstance(messages, list)
        return ModelResponse(text="teacher answer", model="fake-model", metadata={"usage": {"total_tokens": 2}})

def test_teacher_service_generates_with_provenance():
    registry = TeacherRegistry.default()
    service = TeacherService(registry, {"openrouter": FakeProvider()})
    result = service.generate("Explain a loop")
    assert result.response == "teacher answer"
    assert result.teacher == "openrouter"
    assert result.model == "fake-model"
    assert result.metadata["provider"] == "openrouter"
    assert result.metadata["license"] == "provider-output"

def test_teacher_service_rejects_empty_prompt():
    service = TeacherService(TeacherRegistry.default(), {"openrouter": FakeProvider()})
    try:
        service.generate(" ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
