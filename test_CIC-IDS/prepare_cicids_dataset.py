#!/usr/bin/env python
# coding: utf-8
"""
prepare_cicids_dataset.py
=========================
将 CIC-IDS2017 Payload-Bytes 数据转换为作者 Classification_Dataset 期望的格式

输入: Payload_Bytes_File_*.parquet (多个分片)
输出: train.parquet, val.parquet, test.parquet

输出格式:
- question: str, 固定问题
- context: str, 十六进制 payload (每4字符用空格分隔)
- class: int, 数值标签
- type_q: str, 类别名称
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


def bytes_to_hex(byte_array: np.ndarray, format_type: str = 'every4') -> list:
    """
    将字节数组转换为十六进制字符串
    
    Args:
        byte_array: shape (n_samples, n_bytes)
        format_type: 'every4' 或 'every2'
    
    Returns:
        hex_strings: 十六进制字符串列表
    """
    hex_strings = []
    for row in tqdm(byte_array, desc="转换十六进制", leave=False):
        hex_str = ''.join(f'{int(b):02x}' for b in row)
        if format_type == 'every4':
            hex_str = ' '.join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
        elif format_type == 'every2':
            hex_str = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
        hex_strings.append(hex_str.strip())
    return hex_strings


def load_and_merge_parquet(data_dir: Path, pattern: str = "Payload_Bytes_File_*.parquet"):
    """加载并合并所有 parquet 分片"""
    parquet_files = sorted(glob.glob(str(data_dir / pattern)))
    if not parquet_files:
        raise FileNotFoundError(f"未找到匹配 {pattern} 的文件，目录: {data_dir}")
    
    print(f"📂 找到 {len(parquet_files)} 个数据分片")
    
    dfs = []
    for f in tqdm(parquet_files, desc="加载分片"):
        df_part = pd.read_parquet(f)
        dfs.append(df_part)
    
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"✅ 数据加载完成! 总样本数: {len(df_full):,}")
    return df_full


def get_label_column(df: pd.DataFrame) -> str:
    """自动检测标签列"""
    candidates = ['attack_label', 'Label', 'label', 'class']
    for col in candidates:
        if col in df.columns:
            return col
    
    # 尝试模糊匹配
    for col in df.columns:
        if 'label' in col.lower() or 'attack' in col.lower():
            return col
    
    raise ValueError("未找到标签列")


def convert_dataset(
    df: pd.DataFrame,
    max_bytes: int = 64,
    format_type: str = 'every4',
    question: str = "Classify the network packet",
    sample_size: int = None,
    seed: int = 42
):
    """
    将原始数据转换为作者期望的格式
    
    Args:
        df: 原始 DataFrame
        max_bytes: 使用的最大 payload 字节数
        format_type: 十六进制格式 ('every4' 或 'every2')
        question: 固定问题文本
        sample_size: 采样大小 (None 表示使用全部数据)
        seed: 随机种子
    """
    # 获取标签列
    label_col = get_label_column(df)
    print(f"🏷️ 标签列: {label_col}")
    
    # 采样（如有必要）
    if sample_size and sample_size < len(df):
        print(f"📊 分层采样: {sample_size:,} 样本")
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, sample_size // df[label_col].nunique())),
                random_state=seed
            )
        )
        print(f"   采样后: {len(df):,} 样本")
    
    # 获取 payload 列
    payload_cols = sorted(
        [c for c in df.columns if c.startswith('payload_byte_')],
        key=lambda x: int(x.split('_')[-1])
    )[:max_bytes]
    
    if not payload_cols:
        raise ValueError("未找到 payload_byte_* 列")
    
    print(f"📦 使用 {len(payload_cols)} 个 payload 字节列")
    
    # 提取字节数据
    X_bytes = df[payload_cols].values.astype(np.uint8)
    y_labels = df[label_col].values
    
    # 转换为十六进制
    print(f"🔄 转换为十六进制 (格式: {format_type})...")
    contexts = bytes_to_hex(X_bytes, format_type=format_type)
    
    # 创建标签映射
    unique_labels = sorted(df[label_col].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"📋 类别数: {len(unique_labels)}")
    for label, idx in label_to_id.items():
        count = (y_labels == label).sum()
        print(f"   {label} -> {idx} ({count:,} 样本)")
    
    # 构建输出 DataFrame
    result = pd.DataFrame({
        'question': question,
        'context': contexts,
        'class': [label_to_id[label] for label in y_labels],
        'type_q': y_labels
    })
    
    return result, label_to_id


def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42
):
    """划分数据集并保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 划分训练集和临时集
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size),
        stratify=df['class'], random_state=seed
    )
    
    # 划分验证集和测试集
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val_size),
        stratify=temp_df['class'], random_state=seed
    )
    
    # 保存
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ 数据集已保存:")
    print(f"   训练集: {train_path} ({len(train_df):,} 样本)")
    print(f"   验证集: {val_path} ({len(val_df):,} 样本)")
    print(f"   测试集: {test_path} ({len(test_df):,} 样本)")
    
    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(description="转换 CIC-IDS2017 数据为作者格式")
    parser.add_argument(
        "--input_dir", type=str,
        default="../data/CIC-IDS2017/Payload-Bytes",
        help="输入数据目录"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="../data/CIC-IDS2017/Classification",
        help="输出数据目录"
    )
    parser.add_argument(
        "--max_bytes", type=int, default=64,
        help="使用的最大 payload 字节数"
    )
    parser.add_argument(
        "--format", type=str, default="every4",
        choices=["every4", "every2", "noSpace"],
        help="十六进制格式"
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="采样大小 (None 表示使用全部数据)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="仅验证数据格式，不保存"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CIC-IDS2017 数据格式转换")
    print("=" * 60)
    
    # 加载数据
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    df = load_and_merge_parquet(input_dir)
    
    # 转换格式
    result_df, label_map = convert_dataset(
        df,
        max_bytes=args.max_bytes,
        format_type=args.format,
        sample_size=args.sample_size,
        seed=args.seed
    )
    
    # 展示示例
    print(f"\n📝 数据示例:")
    print(result_df.head(3).to_string())
    
    if args.dry_run:
        print("\n🔍 Dry run 模式，跳过保存")
        return
    
    # 划分并保存
    split_and_save(result_df, args.output_dir, seed=args.seed)
    
    # 保存标签映射
    import json
    label_map_path = Path(args.output_dir) / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"   标签映射: {label_map_path}")
    
    print("\n🎉 转换完成!")


if __name__ == "__main__":
    main()
