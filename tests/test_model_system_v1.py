import numpy as np
from core.tensor import Tensor
from models.transformer import Transformer
from cerebro_zero.models import ModelSpec, ModelRegistry, ModelLineage, TeacherRegistry

def test_model_registry_and_lineage():
    registry=ModelRegistry(); model=object(); spec=ModelSpec("cerebro-model","1.0",capabilities=("generation",))
    registry.register(model,spec)
    assert registry.get("cerebro-model") is model
    assert registry.spec("cerebro-model").version=="1.0"
    lineage=ModelLineage("cerebro-model","1.0",teachers=("teacher-a",))
    assert lineage.teachers==("teacher-a",)

def test_transformer_forward_backward_and_cache():
    m=Transformer(vocab_size=20,d_model=16,num_heads=2,d_ff=32,num_layers=1,max_len=8)
    logits=m.forward(Tensor(np.array([[1,2,3,4]]))); logits.sum().backward()
    assert logits.data.shape==(1,4,20)
    assert all(p.grad is not None for p in m.parameters())
    _,cache=m.forward_cached([1],None,0); _,cache=m.forward_cached([2],cache,1)
    assert cache[0]["attention"]["k"].shape[2]==2
