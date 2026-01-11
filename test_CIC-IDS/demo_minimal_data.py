#!/usr/bin/env python
"""
PCAP_encoder 最小化数据演示脚本
================================

此脚本用于生成最小量的模拟网络流量数据，以便快速验证 PCAP_encoder 的工作流程。
无需下载任何外部数据集，所有数据都是本地合成的。

使用方法：
    python demo_minimal_data.py

生成的数据将保存在 ../data/demo/ 目录下。
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def print_banner():
    """打印欢迎横幅"""
    print("=" * 70)
    print("🚀 PCAP_encoder 最小化数据演示脚本")
    print("=" * 70)
    print()
    print("本脚本将生成少量模拟网络流量数据，用于验证 PCAP_encoder 流程。")
    print("无需下载任何大型数据集！")
    print()


def wait_for_user(step: int, total: int, message: str):
    """等待用户按回车继续"""
    print(f"\n{'─' * 70}")
    print(f"[Step {step}/{total}] {message}")
    print("─" * 70)
    input(">>> 按 Enter 键继续...")
    print()


def generate_synthetic_traffic(
    n_samples: int = 50,
    n_bytes: int = 64,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟网络流量数据
    
    数据格式模拟 CIC-IDS2017 Payload-Bytes 数据集：
    - payload_byte_1, payload_byte_2, ..., payload_byte_N: 载荷字节 (0-255)
    - attack_label: 攻击类型标签
    
    Args:
        n_samples: 生成的样本数量
        n_bytes: 每个样本的字节数
        seed: 随机种子
    
    Returns:
        pd.DataFrame: 包含模拟流量数据的 DataFrame
    """
    np.random.seed(seed)
    
    # 定义攻击类型
    attack_types = ["BENIGN", "FTP-Patator", "SSH-Patator"]
    
    # 生成标签（按比例分配）
    labels = np.random.choice(
        attack_types,
        size=n_samples,
        p=[0.6, 0.25, 0.15]  # 60% 正常, 25% FTP攻击, 15% SSH攻击
    )
    
    # 生成载荷字节
    # 根据标签类型生成不同模式的数据，模拟真实差异
    payload_data = []
    for label in labels:
        if label == "BENIGN":
            # 正常流量：较低的字节值，模拟 HTTP/HTTPS
            payload = np.random.randint(0, 128, size=n_bytes)
        elif label == "FTP-Patator":
            # FTP 暴力攻击：包含特定模式
            payload = np.random.randint(32, 127, size=n_bytes)  # ASCII 可打印字符
            payload[:4] = [70, 84, 80, 32]  # "FTP " 的 ASCII
        else:  # SSH-Patator
            # SSH 暴力攻击：包含 SSH 协议特征
            payload = np.random.randint(0, 255, size=n_bytes)
            payload[:4] = [83, 83, 72, 45]  # "SSH-" 的 ASCII
        payload_data.append(payload)
    
    payload_array = np.array(payload_data)
    
    # 构建 DataFrame
    columns = {f"payload_byte_{i+1}": payload_array[:, i] for i in range(n_bytes)}
    columns["attack_label"] = labels
    
    df = pd.DataFrame(columns)
    
    return df


def show_data_summary(df: pd.DataFrame):
    """展示数据摘要"""
    print("📊 数据摘要:")
    print(f"   - 总样本数: {len(df)}")
    print(f"   - 字节列数: {len([c for c in df.columns if c.startswith('payload_byte_')])}")
    print()
    print("📈 标签分布:")
    label_counts = df["attack_label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 5)
        print(f"   {label:15s}: {count:3d} ({pct:5.1f}%) {bar}")
    print()
    print("📋 前 5 条记录 (部分列):")
    display_cols = ["attack_label"] + [f"payload_byte_{i}" for i in range(1, 6)]
    print(df[display_cols].head().to_string(index=True))


def show_hex_conversion(df: pd.DataFrame, n_examples: int = 3):
    """展示字节到十六进制字符串的转换过程"""
    print("🔄 Payload 字节 → 十六进制字符串转换示例:")
    print()
    
    payload_cols = [c for c in df.columns if c.startswith("payload_byte_")][:16]  # 只取前16字节
    
    for i in range(min(n_examples, len(df))):
        row = df.iloc[i]
        label = row["attack_label"]
        bytes_data = [int(row[col]) for col in payload_cols]
        
        # 原始字节
        bytes_str = " ".join(f"{b:3d}" for b in bytes_data[:8])
        
        # 十六进制字符串
        hex_str = "".join(f"{b:02x}" for b in bytes_data)
        
        # 每4字符分组（PCAP_encoder 的 "every4" 格式）
        hex_grouped = " ".join(hex_str[j:j+4] for j in range(0, len(hex_str), 4))
        
        print(f"   样本 {i+1} ({label}):")
        print(f"   原始字节 (前8个): [{bytes_str} ...]")
        print(f"   十六进制字符串:   {hex_grouped}")
        print()


def save_as_parquet(df: pd.DataFrame, output_dir: Path) -> Path:
    """保存数据为 Parquet 格式"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_payload_bytes.parquet"
    
    df.to_parquet(output_path, index=False)
    
    # 计算文件大小
    size_bytes = output_path.stat().st_size
    size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes > 1024 else f"{size_bytes} bytes"
    
    print(f"💾 数据已保存:")
    print(f"   路径: {output_path}")
    print(f"   大小: {size_str}")
    
    return output_path


def show_next_steps(parquet_path: Path):
    """展示后续步骤"""
    print("🎯 后续步骤:")
    print()
    print("   1. 查看数据标签分布:")
    print(f"      python inspect_labels.py --data {parquet_path}")
    print()
    print("   2. 运行端到端演示:")
    print("      python demo_pipeline.py")
    print()
    print("   3. 使用编码器+分类头评估 (需要预训练权重):")
    print(f"      python eval_with_encoder_head.py --data {parquet_path} --sample 50")
    print()


def main():
    """主函数"""
    print_banner()
    
    # 确定输出目录
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = repo_root / "data" / "demo"
    
    # Step 1: 生成合成数据
    wait_for_user(1, 4, "生成模拟网络流量数据")
    
    n_samples = 50  # 极小数据量
    n_bytes = 64    # 每个样本的字节数
    
    print(f"⚙️  配置:")
    print(f"   - 样本数量: {n_samples}")
    print(f"   - 每样本字节数: {n_bytes}")
    print()
    
    df = generate_synthetic_traffic(n_samples=n_samples, n_bytes=n_bytes)
    print("✅ 数据生成完成!")
    print()
    
    # Step 2: 展示数据摘要
    wait_for_user(2, 4, "查看数据摘要和标签分布")
    show_data_summary(df)
    
    # Step 3: 展示转换过程
    wait_for_user(3, 4, "查看字节到十六进制的转换过程")
    show_hex_conversion(df)
    
    # Step 4: 保存数据
    wait_for_user(4, 4, "保存数据为 Parquet 格式")
    parquet_path = save_as_parquet(df, output_dir)
    print()
    
    # 完成
    print("=" * 70)
    print("🎉 演示完成!")
    print("=" * 70)
    print()
    show_next_steps(parquet_path)


if __name__ == "__main__":
    main()
