# PCAP_encoder 项目规则

## Python 环境
- 所有 Python 命令必须使用 `myenv` 虚拟环境
- 命令格式：`source ~/myenv/bin/activate && python3 <script>`
- 或者直接使用：`~/myenv/bin/python3 <script>`

## 依赖管理
- 使用 `~/myenv/bin/pip` 安装依赖
- 已知依赖：pandas, numpy, torch, transformers, pyarrow, scapy

## 数据路径
- 测试数据存放在：`../data/demo/`
- 真实数据存放在：`../data/CIC-IDS2017/`
