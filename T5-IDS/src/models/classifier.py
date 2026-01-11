"""Classification head scaffold."""

from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    input_dim: int = 768
    num_classes: int = 2


class PacketClassifier:
    def __init__(self, config: ClassifierConfig):
        self.config = config

    def forward(self, embeddings):
        """Return class logits."""
        raise NotImplementedError
