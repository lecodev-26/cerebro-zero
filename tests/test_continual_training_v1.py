import numpy as np
from models.transformer import Transformer
from cerebro_zero.training.continual import (
    GROWTH_SCHEDULE,
    GrowthStage,
    estimate_parameters,
    expand_from_checkpoint,
    stage_for_generation,
)


def test_growth_schedule_is_monotonic():
    counts = [estimate_parameters(512, 128, s) for s in GROWTH_SCHEDULE]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_same_lineage_generation_grows_capacity():
    a = stage_for_generation(0)
    b = stage_for_generation(1)
    assert b.generation == a.generation + 1
    assert b.d_model > a.d_model
    assert b.d_ff > a.d_ff


def test_growth_can_continue_after_schedule():
    s = stage_for_generation(7)
    assert s.generation == 7
    assert s.d_model > GROWTH_SCHEDULE[-1].d_model


def test_expansion_preserves_parent_weight_overlap():
    np.random.seed(7)
    parent = Transformer(vocab_size=32, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=8)
    old = [p.data.copy() for p in parent.parameters()]
    stage = GrowthStage(1, 12, 2, 24, 2)
    child = expand_from_checkpoint(parent, {}, stage)
    assert child.num_parameters() > parent.num_parameters()
    for src, dst in zip(old, child.parameters()):
        slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.data.shape, src.shape))
        assert np.allclose(dst.data[slices], src[slices])
