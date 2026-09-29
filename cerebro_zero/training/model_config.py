from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class CerebroModelConfig:
    """Frozen, reproducible starting configuration for Cerebro Model 1.0."""
    version: str = "1.0"
    family: str = "cerebro"
    architecture: str = "decoder-transformer"
    tokenizer: str = "byte-bpe"
    vocab_size: int = 512
    context_length: int = 128
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    num_layers: int = 4
    learning_rate: float = 3e-4
    batch_size: int = 8
    epochs: int = 3
    batches_per_epoch: int = 200
    seed: int = 42
    min_quality: float = 0.80
    train_fraction: float = 0.90
    validation_fraction: float = 0.05
    test_fraction: float = 0.05

    def validate(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("dataset fractions must sum to 1")
        if self.vocab_size < 260:
            raise ValueError("vocab_size must be >= 260")
        return self

    def to_dict(self):
        return asdict(self)
