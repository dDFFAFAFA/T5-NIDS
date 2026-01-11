"""Training loop scaffold."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 2e-5
    seed: int = 42
    device: str = "cuda"


class Trainer:
    def __init__(self, config: TrainConfig, logger=None):
        self.config = config
        self.logger = logger

    def train(self, model, dataloader):
        """Run training loop."""
        raise NotImplementedError

    def evaluate(self, model, dataloader) -> Dict[str, Any]:
        """Return evaluation metrics."""
        raise NotImplementedError
