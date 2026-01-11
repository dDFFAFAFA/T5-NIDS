# T5-IDS

Lightweight scaffold for PCAP-based NIDS experiments. This directory is isolated
from the original codebase to keep your changes clean and easy to find.

## Layout
- config/: default configuration values
- src/: reusable modules (data, models, training, utils)
- experiments/: run configs and notes (pretrain, finetune)
- notebooks/: interactive demos
- data/: path conventions only (no raw data committed)
- checkpoints/: weights and training checkpoints
- results/: metrics, reports, and figures

## Quick Start
1. Place data under `data/raw/` or update paths in `config/default_config.py`.
2. Use `notebooks/NIDS_Pipeline.ipynb` as a guided demo entry point.
