"""T5 encoder + bottleneck scaffold."""

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    model_name: str = "t5-base"
    bottleneck: str = "mean"
    pkt_repr_dim: int = 768
    use_pkt_reduction: bool = False


class T5EncoderWithBottleneck:
    def __init__(self, config: EncoderConfig):
        self.config = config

    def forward(self, input_ids, attention_mask):
        """Return packet representations."""
        raise NotImplementedError
