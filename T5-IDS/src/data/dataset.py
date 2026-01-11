"""Dataset scaffold for NIDS tasks."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DatasetConfig:
    input_format: str = "every4"
    question_template: str = "Classify the network packet"


class NIDSDataset:
    def __init__(self, records: List[Dict[str, Any]], config: DatasetConfig):
        self.records = records
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]
