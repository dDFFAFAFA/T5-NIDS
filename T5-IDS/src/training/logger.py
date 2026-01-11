"""Experiment logging scaffold."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class LogConfig:
    results_dir: Path


class ExperimentLogger:
    def __init__(self, config: LogConfig):
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Persist metrics (implement as needed)."""
        pass

    def save_artifact(self, name: str, data: bytes):
        """Save binary artifacts like checkpoints."""
        path = self.config.results_dir / name
        path.write_bytes(data)
