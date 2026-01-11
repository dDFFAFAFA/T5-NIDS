#!/usr/bin/env python
# coding: utf-8
"""
run_cicids_training.py
======================
CIC-IDS2017 分类训练的 Python 入口脚本
可以直接运行，无需 shell 脚本

用法:
    python run_cicids_training.py [--unfreeze] [--lr 0.00001]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="CIC-IDS2017 NIDS 分类训练")
    parser.add_argument(
        "--unfreeze", action="store_true",
        help="解冻编码器进行微调"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="学习率 (默认: 冻结=0.001, 解冻=0.00001)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=24,
        help="批次大小"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--percentage", type=int, default=100,
        help="使用数据的百分比 [1, 100]"
    )
    parser.add_argument(
        "--loss", type=str, default="normal",
        choices=["normal", "weighted"],
        help="损失函数类型"
    )
    parser.add_argument(
        "--bottleneck", type=str, default="mean",
        choices=["mean", "first", "last", "attention"],
        help="Bottleneck 策略"
    )
    
    args = parser.parse_args()
    
    # 设置学习率
    if args.lr is None:
        args.lr = 0.00001 if args.unfreeze else 0.001
    
    # 路径配置
    data_dir = PROJECT_ROOT / "data" / "CIC-IDS2017" / "Classification"
    model_path = PROJECT_ROOT / "models" / "pretrained"
    script_path = PROJECT_ROOT / "2.Training" / "classification" / "classification.py"
    output_path = PROJECT_ROOT / "results"
    
    # 检查数据文件
    train_file = data_dir / "train.parquet"
    val_file = data_dir / "val.parquet"
    test_file = data_dir / "test.parquet"
    
    if not train_file.exists():
        print(f"❌ 训练数据不存在: {train_file}")
        print("   请先运行 prepare_cicids_dataset.py 生成数据:")
        print(f"   python test_CIC-IDS/prepare_cicids_dataset.py --input_dir <your_data_dir>")
        sys.exit(1)
    
    # 构建实验标识符
    encoder_status = "unfrozen" if args.unfreeze else "frozen"
    identifier = f"cicids_lr{args.lr}_seed{args.seed}_loss{args.loss}_batch{args.batch_size}_{encoder_status}"
    
    # 构建命令行参数
    cmd_args = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes=1",
        str(script_path),
        "--identifier", identifier,
        "--experiment", "CIC-IDS2017_NIDS",
        "--task", "supervised",
        "--clean_start",
        "--tokenizer_name", "T5-base",
        "--model_name", "T5-base",
        "--finetuned_path_model", str(model_path),
        "--training_data", str(train_file),
        "--validation_data", str(val_file),
        "--testing_data", str(test_file),
        "--output_path", str(output_path),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--percentage", str(args.percentage),
        "--loss", args.loss,
        "--bottleneck", args.bottleneck,
        "--max_qst_length", "512",
        "--max_ans_length", "32",
        "--log_level", "info",
        "--gpu", "0,",
    ]
    
    # 添加冻结参数
    if not args.unfreeze:
        cmd_args.append("--fix_encoder")
    
    # 打印配置
    print("=" * 60)
    print("🚀 CIC-IDS2017 NIDS 分类训练")
    print("=" * 60)
    print(f"📋 实验标识符: {identifier}")
    print(f"📂 训练数据: {train_file}")
    print(f"🔧 学习率: {args.lr}")
    print(f"🔧 Epochs: {args.epochs}")
    print(f"🔧 Batch Size: {args.batch_size}")
    print(f"🔧 编码器状态: {encoder_status}")
    print(f"🔧 Bottleneck: {args.bottleneck}")
    print("=" * 60)
    
    # 运行训练
    print("\n🏃 启动训练...")
    os.chdir(PROJECT_ROOT)
    result = subprocess.run(cmd_args)
    
    if result.returncode == 0:
        print("\n✅ 训练完成!")
    else:
        print(f"\n❌ 训练失败，返回码: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
