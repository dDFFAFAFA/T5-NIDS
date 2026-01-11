#!/usr/bin/env python
"""
此脚本用于评估预训练的 T5 编码器（weights.pth）在 CIC-IDS2017 数据集上的表现。
主要流程：
1. 加载包含 Payload 字节和攻击标签的 Parquet 文件。
2. 将 Payload 字节转换为十六进制字符串（Context）。
3. 使用 T5 分词器对 (Question, Context) 进行编码。
4. 加载预训练的 T5 编码器并冻结其参数（不参与训练）。
5. 在编码器之上添加一个简单的线性分类头（Linear Head）。
6. 训练该分类头，并计算 F1-score 和 AUC 等指标。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用冻结编码器+线性头评估 CIC-IDS")
    parser.add_argument("--data", type=Path, default=Path("../data/CIC-IDS2017/Payload-Bytes/Payload_Bytes_File_1.parquet"), help="包含 payload_byte_* 和 attack_label 的 Parquet 文件路径")
    parser.add_argument("--weights", type=Path, default=Path("../models/weights.pth"), help="预训练编码器权重 weights.pth 的路径")
    parser.add_argument("--model-name", type=str, default="t5-base", help="HuggingFace 基础模型名称")
    parser.add_argument("--question", type=str, default="Classify the network packet", help="用于分词的问题前缀")
    parser.add_argument("--max-bytes", type=int, default=512, help="使用的 payload_byte_* 列的最大数量")
    parser.add_argument("--max-length", type=int, default=512, help="分词器的最大 Token 长度")
    parser.add_argument("--sample", type=int, default=8000, help="可选的数据下采样行数；0 表示使用全部数据")
    parser.add_argument("--train-frac", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-frac", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--batch-size", type=int, default=8, help="训练和评估的 Batch Size")
    parser.add_argument("--epochs", type=int, default=3, help="线性分类头的训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="分类头的学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备 (cuda 或 cpu)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子以保证实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_payload_df(path: Path, max_bytes: int, sample: int, seed: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载 Parquet 数据并进行预处理。
    注意：该数据集的 Payload 字节列通常从 payload_byte_1 开始。
    """
    # 动态获取可用的 payload 列名，避免请求不存在的列（如 payload_byte_0）
    pf = pq.ParquetFile(path)
    payload_cols_all = sorted(
        [c for c in pf.schema.names if c.startswith("payload_byte_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    # 仅取前 max_bytes 个字节列，减少内存占用
    payload_cols = payload_cols_all[:max_bytes]
    cols = ["attack_label"] + payload_cols
    
    print(f"正在从 {path} 读取数据...")
    df = pd.read_parquet(path, columns=cols)
    # 剔除标签缺失的行
    df = df.dropna(subset=["attack_label"])
    # 填充缺失值并转换为 uint16 以节省空间
    df[payload_cols] = df[payload_cols].fillna(0).astype(np.uint16)
    
    # 如果指定了采样数量，则进行随机下采样
    if sample and sample > 0 and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
        
    return df.reset_index(drop=True), payload_cols


def build_text_fields(df: pd.DataFrame, payload_cols: List[str], question: str) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """
    将原始字节数据转换为模型可理解的文本字段。
    """
    # 将每行字节数组转换为十六进制字符串（Context）
    byte_array = df[payload_cols].to_numpy(dtype=np.uint16)
    contexts = ["".join(f"{int(b):02x}" for b in row) for row in byte_array]
    # 构造问题列表
    questions = [question] * len(contexts)
    # 将字符串标签映射为整数 ID
    labels, uniques = pd.factorize(df["attack_label"], sort=True)
    return questions, contexts, labels.astype(np.int64), uniques.tolist()


def tokenize(tokenizer: T5TokenizerFast, questions: List[str], contexts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    """
    使用 T5 分词器对文本进行编码。
    """
    encoded = tokenizer(
        questions,
        contexts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask}


def make_loaders(encodings: Dict[str, torch.Tensor], labels: np.ndarray, batch_size: int, seed: int, train_frac: float, val_frac: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试集的 DataLoader。
    使用分层抽样 (stratify) 以保持各集合中类别比例一致。
    """
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("训练集和验证集比例之和必须小于 1.0")
        
    idx = np.arange(len(labels))
    # 划分训练集和临时集
    train_idx, temp_idx = train_test_split(idx, test_size=(val_frac + test_frac), stratify=labels, random_state=seed)
    # 从临时集中划分验证集和测试集
    val_ratio = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1 - val_ratio), stratify=labels[temp_idx], random_state=seed)

    def _loader(subset_idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            encodings["input_ids"][subset_idx],
            encodings["attention_mask"][subset_idx],
            torch.tensor(labels[subset_idx], dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)


def load_encoder(model_name: str, weights_path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    """
    加载 T5 模型，应用预训练权重，并提取冻结的编码器。
    """
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # 加载自定义权重 (weights.pth)
    try:
        # 尝试使用 weights_only=True 以提高安全性
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
        
    # 加载权重，允许不完全匹配 (strict=False)
    missing = model.load_state_dict(state, strict=False)
    print(f"已加载权重。缺失的键: {missing.missing_keys}")
    
    # 提取编码器部分并移动到指定设备
    encoder = model.encoder.to(device)
    # 冻结编码器参数，使其在训练过程中不更新
    for p in encoder.parameters():
        p.requires_grad = False
        
    return encoder, model.config.d_model


def run_epoch(loader: DataLoader, encoder: nn.Module, head: nn.Module, device: torch.device, train: bool, optimizer=None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    运行一个训练或评估周期。
    """
    preds, targets = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    if train:
        head.train()
    else:
        head.eval()
        
    for batch in loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        
        with torch.set_grad_enabled(train):
            # 1. 编码器前向传播
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            # 取第一个 Token (通常代表全局特征) 的隐藏状态
            hidden = outputs.last_hidden_state[:, 0, :]
            # 2. 分类头前向传播
            logits = head(hidden)
            # 3. 计算损失
            loss = criterion(logits, labels)
            
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
        total_loss += loss.item() * labels.size(0)
        preds.append(logits.detach().cpu())
        targets.append(labels.detach().cpu())
        
    # 计算概率分布
    probs = torch.softmax(torch.cat(preds, dim=0), dim=1).numpy()
    y_true = torch.cat(targets, dim=0).numpy()
    return probs, y_true, total_loss / len(y_true)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. 加载数据
    df, payload_cols = load_payload_df(args.data, args.max_bytes, args.sample, args.seed)
    print(f"已加载 {len(df)} 行数据。使用 {len(payload_cols)} 个字节列。")

    # 2. 准备文本和标签
    questions, contexts, labels, label_names = build_text_fields(df, payload_cols, args.question)
    
    # 3. 分词
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    encodings = tokenize(tokenizer, questions, contexts, args.max_length)

    # 4. 创建数据加载器
    train_loader, val_loader, test_loader = make_loaders(encodings, labels, args.batch_size, args.seed, args.train_frac, args.val_frac)
    num_labels = len(np.unique(labels))

    # 5. 初始化模型
    encoder, hidden_size = load_encoder(args.model_name, args.weights, device)
    # 仅训练这个线性层
    head = nn.Linear(hidden_size, num_labels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    # 6. 训练循环
    best_val_f1 = -1.0
    print("\n开始训练分类头...")
    for epoch in range(args.epochs):
        train_probs, train_true, train_loss = run_epoch(train_loader, encoder, head, device, train=True, optimizer=optimizer)
        val_probs, val_true, val_loss = run_epoch(val_loader, encoder, head, device, train=False)
        
        val_pred = val_probs.argmax(axis=1)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        print(f"Epoch {epoch+1}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 验证 F1={val_f1:.4f}")
        
        # 保存最佳模型权重
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = head.state_dict()
            
    if best_val_f1 >= 0:
        head.load_state_dict(best_state)

    # 7. 最终测试
    print("\n正在测试集上进行最终评估...")
    test_probs, test_true, test_loss = run_epoch(test_loader, encoder, head, device, train=False)
    test_pred = test_probs.argmax(axis=1)
    
    f1_macro = f1_score(test_true, test_pred, average="macro")
    f1_micro = f1_score(test_true, test_pred, average="micro")
    
    # 计算多分类 AUC (One-vs-Rest)
    try:
        auc_ovr = roc_auc_score(test_true, test_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_ovr = float('nan') # 如果测试集中只有一类，则无法计算 AUC

    # 8. 输出结果
    metrics = {
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "auc_ovr": float(auc_ovr),
        "test_loss": float(test_loss),
        "label_mapping": {int(i): name for i, name in enumerate(label_names)},
    }
    print("\n评估结果:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

