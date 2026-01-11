"""Default configuration for T5-IDS."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Paths:
    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    checkpoints: Path = PROJECT_ROOT / "checkpoints"
    results: Path = PROJECT_ROOT / "results"


@dataclass
class DataConfig:
    question_template: str = "Classify the network packet"
    payload_max_bytes: int = 512
    input_format: str = "every4"


@dataclass
class TrainConfig:
    model_name: str = "t5-base"
    tokenizer_name: str = "t5-base"
    bottleneck: str = "mean"
    pkt_repr_dim: int = 768
    use_pkt_reduction: bool = False
    max_qst_length: int = 512
    max_ans_length: int = 32
    batch_size: int = 16
    epochs: int = 10
    lr: float = 2e-5
    seed: int = 42
    device: str = "cuda"


def build_default_config() -> dict:
    return {
        "paths": Paths(),
        "data": DataConfig(),
        "train": TrainConfig(),
    }
