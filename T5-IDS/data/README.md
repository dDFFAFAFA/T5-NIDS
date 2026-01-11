# Data Directory

This folder only stores path conventions. Do not commit raw datasets.

- raw/: original PCAP or parquet shards
- processed/: derived datasets for training and finetuning

Update `config/default_config.py` to point to external storage if needed.
