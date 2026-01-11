# Repository Guidelines

## Project Structure & Module Organization
- `Core/` contains shared classes, datasets, and utility functions used by training and evaluation.
- `Preprocess/` holds scripts that convert raw PCAP files into parquet datasets.
- `2.Training/` is the main training entry point for QA, denoiser, and classification stages.
- `Experiments/` provides shell scripts with Accelerate configs and hyperparameters.
- `test_CIC-IDS/` contains demo, evaluation, and data-prep scripts for CIC-IDS workflows.
- `data/` stores small demo assets; large datasets are referenced outside the repo.
- `models/` stores pretrained weights (for example `models/weights.pth`).
- `docs/` and `notebooks/` document pipelines and runnable examples.

## Build, Test, and Development Commands
- `conda create -n pcapencoder -f environment.yml` sets up the Python 3.10 environment.
- Local automation expects `~/myenv`; run scripts as `~/myenv/bin/python3 <script>`.
- Demo data: `~/myenv/bin/python3 test_CIC-IDS/demo_minimal_data.py`.
- End-to-end demo: `~/myenv/bin/python3 test_CIC-IDS/demo_pipeline.py --use-pretrained --n-samples 20`.
- Evaluation: `~/myenv/bin/python3 test_CIC-IDS/eval_with_encoder_head.py --data ../data/demo/demo_payload_bytes.parquet --weights models/weights.pth`.
- Training runs via Accelerate, for example `bash Experiments/4_QA_model_training/T5QandA.sh`.

## Coding Style & Naming Conventions
- Use 4-space indentation and follow the existing module layout in `Core/` and `2.Training/`.
- Prefer `snake_case` for functions/variables and `CapWords` for classes.
- Keep dataset column patterns like `payload_byte_1` and config variables uppercase in `Experiments/*.sh`.

## Testing Guidelines
- There is no pytest harness or coverage target; validation is script-driven.
- Follow existing naming patterns (`demo_*.py`, `eval_*.py`, `prepare_*.py`) for new checks.
- Smoke-test changes with `test_CIC-IDS/demo_minimal_data.py` and `test_CIC-IDS/demo_pipeline.py`.

## Commit & Pull Request Guidelines
- This checkout has no git history; use short imperative subjects (for example "Add denoiser dataset options").
- PRs should include a brief summary, commands run, data paths used, and any metrics or logs produced.
- Note when weights or datasets live outside the repo or require downloads.

## Data & Configuration Notes
- Demo data lives in `data/demo/`; CIC-IDS data is expected under `../data/CIC-IDS2017/`.
- New datasets or weight artifacts should be documented in `docs/` or `README.md`.
