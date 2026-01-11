"""Tokenizer wrapper scaffold."""

from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    name: str = "t5-base"
    max_qst_length: int = 512
    max_ans_length: int = 32


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config

    def encode(self, question: str, context: str):
        """Return tokenized inputs for the model."""
        raise NotImplementedError
