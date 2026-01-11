#!/usr/bin/env python
"""
CIC-IDS2017 Payload-Bytes Parquet 数据的快速检查脚本：
1. 打印标签分布（前 N 个）。
2. 显示存在的 payload_byte_* 列及其索引范围。
3. 可选地进行下采样以提高处理速度。
"""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="检查标签和 Payload 列")
    parser.add_argument("--data", type=Path, default=Path("../data/CIC-IDS2017/Payload-Bytes/Payload_Bytes_File_1.parquet"), help="Parquet 文件路径")
    parser.add_argument("--top", type=int, default=20, help="要显示的标签数量（前 N 个）")
    parser.add_argument("--sample", type=int, default=0, help="可选的数据下采样行数；0 表示使用全部数据")
    return parser.parse_args()


def main():
    """主函数：读取 Parquet 文件并打印统计信息"""
    args = parse_args()
    path = args.data
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    # 1. 检查 Schema（元数据）
    # 读取 Parquet 文件的元数据，只统计 payload_byte_* 的范围，避免加载全部数据。
    pf = pq.ParquetFile(path)
    payload_cols = sorted([c for c in pf.schema.names if c.startswith("payload_byte_")], key=lambda x: int(x.split("_")[-1]))
    print(f"总列数: {len(pf.schema.names)}")
    print(f"Payload 字节列数: {len(payload_cols)} (起始列={payload_cols[0]}, 结束列={payload_cols[-1]})")

    # 2. 检查标签分布
    cols = ["attack_label"]
    # 仅拉取 attack_label 列，必要时下采样提升速度。
    df = pd.read_parquet(path, columns=cols)
    if args.sample and len(df) > args.sample:
        print(f"正在下采样至 {args.sample} 行...")
        df = df.sample(n=args.sample, random_state=42)
    
    vc = df["attack_label"].value_counts()
    print(f"使用的总行数: {len(df)}")
    print("\n标签分布 (前 N 个):")
    print("-" * 40)
    print(vc.head(args.top))
    print("-" * 40)


if __name__ == "__main__":
    main()

