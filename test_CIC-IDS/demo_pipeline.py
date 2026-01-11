#!/usr/bin/env python
"""
PCAP_encoder 端到端演示脚本
===========================

此脚本演示 PCAP_encoder 的完整工作流程：
1. 数据准备：加载或生成测试数据
2. 文本构建：将字节数据转换为十六进制上下文
3. 分词编码：使用 T5 分词器处理输入
4. 模型推理：通过 T5 编码器获取表示
5. 分类预测：使用线性分类头进行预测

使用方法：
    python demo_pipeline.py [--use-pretrained]

选项：
    --use-pretrained  使用预训练权重（需要 weights.pth 文件）
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def print_banner():
    """打印欢迎横幅"""
    print("=" * 70)
    print("🔬 PCAP_encoder 端到端演示脚本")
    print("=" * 70)
    print()
    print("本脚本将演示完整的网络流量表示学习流程。")
    print("每个步骤都会详细展示输入和输出。")
    print()


def wait_for_user(step: int, total: int, message: str):
    """等待用户按回车继续"""
    print(f"\n{'─' * 70}")
    print(f"[Step {step}/{total}] {message}")
    print("─" * 70)
    input(">>> 按 Enter 键继续...")
    print()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PCAP_encoder 端到端演示")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        default=True,  # 默认使用预训练权重
        help="使用预训练权重（默认启用，需要 models/weights.pth）"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不使用预训练权重，使用随机初始化"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="数据文件路径（Parquet 格式）"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="演示使用的样本数量"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,  # 自动检测：有预训练权重用 t5-base，否则用 t5-small
        help="T5 模型名称（默认自动检测：预训练用 t5-base，演示用 t5-small）"
    )
    args = parser.parse_args()
    # 处理互斥选项
    if args.no_pretrained:
        args.use_pretrained = False
    
    # 自动检测模型名称
    if args.model_name is None:
        script_dir = Path(__file__).resolve().parent
        weights_path = script_dir.parent / "models" / "weights.pth"
        if args.use_pretrained and weights_path.exists():
            # PCAP_encoder 预训练权重基于 t5-base (隐藏维度=768, 12层)
            args.model_name = "t5-base"
        else:
            args.model_name = "t5-small"
    
    return args


def check_dependencies() -> bool:
    """检查必要的依赖是否已安装"""
    print("🔍 检查依赖...")
    
    missing = []
    
    try:
        import transformers
        print(f"   ✅ transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
        print("   ❌ transformers 未安装")
    
    try:
        import torch
        print(f"   ✅ torch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("   ❌ torch 未安装")
    
    try:
        import pandas
        print(f"   ✅ pandas {pandas.__version__}")
    except ImportError:
        missing.append("pandas")
        print("   ❌ pandas 未安装")
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    return True


def load_or_generate_data(data_path: Optional[Path], n_samples: int) -> pd.DataFrame:
    """加载现有数据或生成演示数据"""
    
    if data_path and data_path.exists():
        print(f"📂 从文件加载数据: {data_path}")
        df = pd.read_parquet(data_path)
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
        print(f"   已加载 {len(df)} 条记录")
    else:
        print("🔧 生成演示数据...")
        # 生成简单的演示数据
        np.random.seed(42)
        
        attack_types = ["BENIGN", "FTP-Patator", "SSH-Patator"]
        labels = np.random.choice(attack_types, size=n_samples, p=[0.5, 0.3, 0.2])
        
        # 生成 32 字节的载荷
        payload_data = np.random.randint(0, 256, size=(n_samples, 32))
        
        columns = {f"payload_byte_{i+1}": payload_data[:, i] for i in range(32)}
        columns["attack_label"] = labels
        
        df = pd.DataFrame(columns)
        print(f"   已生成 {len(df)} 条演示记录")
    
    return df


def build_text_fields(df: pd.DataFrame, question: str = "Classify the network packet") -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """
    将数据转换为模型输入格式
    
    Returns:
        questions: 问题列表
        contexts: 上下文列表（十六进制字符串）
        labels: 标签数组
        label_names: 标签名称列表
    """
    print("🔄 将载荷字节转换为十六进制上下文...")
    
    # 获取所有 payload 列
    payload_cols = sorted(
        [c for c in df.columns if c.startswith("payload_byte_")],
        key=lambda x: int(x.split("_")[-1])
    )
    
    # 转换为十六进制字符串
    byte_array = df[payload_cols].to_numpy(dtype=np.uint16)
    contexts = []
    for row in byte_array:
        hex_str = "".join(f"{int(b):02x}" for b in row)
        # 每4字符分组（PCAP_encoder 格式）
        hex_grouped = " ".join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
        contexts.append(hex_grouped)
    
    # 构造问题
    questions = [question] * len(contexts)
    
    # 标签编码
    labels, uniques = pd.factorize(df["attack_label"], sort=True)
    
    print(f"   问题模板: \"{question}\"")
    print(f"   上下文长度: {len(contexts[0])} 字符")
    print(f"   标签映射: {dict(enumerate(uniques))}")
    
    return questions, contexts, labels.astype(np.int64), uniques.tolist()


def show_sample_conversion(df: pd.DataFrame, questions: List[str], contexts: List[str], n: int = 2):
    """展示样本转换的详细过程"""
    print("📋 样本转换详情:")
    print()
    
    payload_cols = sorted(
        [c for c in df.columns if c.startswith("payload_byte_")],
        key=lambda x: int(x.split("_")[-1])
    )[:8]  # 只取前8字节
    
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        print(f"   ┌─ 样本 {i+1} ─────────────────────────────────────────────")
        print(f"   │ 标签: {row['attack_label']}")
        
        bytes_vals = [int(row[col]) for col in payload_cols]
        print(f"   │ 原始字节 (前8): {bytes_vals}")
        
        print(f"   │ 问题: {questions[i][:50]}...")
        print(f"   │ 上下文: {contexts[i][:60]}...")
        print(f"   └{'─' * 55}")
        print()


def tokenize_inputs(
    questions: List[str],
    contexts: List[str],
    tokenizer,
    max_length: int = 128
) -> Dict[str, torch.Tensor]:
    """使用 T5 分词器编码输入"""
    print("🔤 分词编码...")
    
    # T5 的输入格式：question + context
    inputs = [f"question: {q} context: {c}" for q, c in zip(questions, contexts)]
    
    encoded = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    print(f"   输入序列数: {len(inputs)}")
    print(f"   Token 序列形状: {encoded['input_ids'].shape}")
    print(f"   最大序列长度: {encoded['input_ids'].shape[1]}")
    
    # 展示第一个样本的分词结果
    print()
    print("   第一个样本的分词结果:")
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0][:20])
    print(f"   Tokens (前20): {tokens}")
    
    return encoded


def load_model(model_name: str, weights_path: Optional[Path], device: str) -> Tuple[nn.Module, int]:
    """加载 T5 编码器"""
    from transformers import T5ForConditionalGeneration
    
    print(f"🤖 加载 T5 模型: {model_name}")
    
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    if weights_path and weights_path.exists():
        print(f"   加载预训练权重: {weights_path}")
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("   ✅ 权重加载成功")
    else:
        print("   ⚠️  使用随机初始化权重（仅用于演示流程）")
    
    encoder = model.encoder.to(device)
    
    # 冻结参数
    for p in encoder.parameters():
        p.requires_grad = False
    
    hidden_size = model.config.d_model
    print(f"   编码器隐藏维度: {hidden_size}")
    print(f"   编码器层数: {model.config.num_layers}")
    print(f"   设备: {device}")
    
    return encoder, hidden_size


def encode_and_classify(
    encoder: nn.Module,
    head: nn.Module,
    encodings: Dict[str, torch.Tensor],
    labels: np.ndarray,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """通过编码器和分类头进行推理"""
    print("🧠 编码 + 分类推理...")
    
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    with torch.no_grad():
        # 编码
        print("   1. 通过 T5 编码器...")
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state[:, 0, :]  # 取第一个 token 的表示
        print(f"      隐藏表示形状: {hidden.shape}")
        
        # 分类
        print("   2. 通过线性分类头...")
        logits = head(hidden)
        print(f"      Logits 形状: {logits.shape}")
        
        # 预测
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    
    return preds, probs


def show_predictions(
    df: pd.DataFrame,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    label_names: List[str]
):
    """展示预测结果"""
    print("📊 预测结果:")
    print()
    print(f"   {'样本':^6} │ {'真实标签':^15} │ {'预测标签':^15} │ {'置信度':^10} │ {'正确':^6}")
    print(f"   {'─'*6}─┼─{'─'*15}─┼─{'─'*15}─┼─{'─'*10}─┼─{'─'*6}")
    
    correct = 0
    for i in range(len(labels)):
        true_label = label_names[labels[i]]
        pred_label = label_names[preds[i]]
        confidence = probs[i].max() * 100
        is_correct = "✅" if labels[i] == preds[i] else "❌"
        if labels[i] == preds[i]:
            correct += 1
        
        print(f"   {i+1:^6} │ {true_label:^15} │ {pred_label:^15} │ {confidence:^10.1f}% │ {is_correct:^6}")
    
    accuracy = correct / len(labels) * 100
    print(f"   {'─'*6}─┴─{'─'*15}─┴─{'─'*15}─┴─{'─'*10}─┴─{'─'*6}")
    print(f"\n   准确率: {accuracy:.1f}% ({correct}/{len(labels)})")
    
    # 注意事项
    print()
    print("   📝 注意:")
    print("   - 如果未使用预训练权重，准确率接近随机猜测是正常的")
    print("   - 这只是演示流程，不代表真实模型性能")


def main():
    """主函数"""
    args = parse_args()
    
    print_banner()
    
    # Step 0: 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 导入（在依赖检查后）
    from transformers import T5TokenizerFast
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  运行设备: {device}")
    
    # Step 1: 加载/生成数据
    wait_for_user(1, 5, "加载或生成测试数据")
    df = load_or_generate_data(args.data, args.n_samples)
    print()
    print(df.head())
    
    # Step 2: 构建文本字段
    wait_for_user(2, 5, "将字节数据转换为文本格式")
    questions, contexts, labels, label_names = build_text_fields(df)
    print()
    show_sample_conversion(df, questions, contexts)
    
    # Step 3: 分词
    wait_for_user(3, 5, "使用 T5 分词器编码")
    print(f"⏳ 正在加载分词器 ({args.model_name})...")
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    encodings = tokenize_inputs(questions, contexts, tokenizer)
    
    # Step 4: 加载模型
    wait_for_user(4, 5, "加载 T5 编码器和分类头")
    
    # 默认尝试加载预训练权重
    script_dir = Path(__file__).resolve().parent
    weights_path = script_dir.parent / "models" / "weights.pth"
    
    if not args.use_pretrained:
        # 用户明确不使用预训练权重
        print("⚠️  用户选择不使用预训练权重")
        weights_path = None
    elif not weights_path.exists():
        print(f"⚠️  未找到权重文件: {weights_path}")
        print("   将使用随机初始化权重")
        weights_path = None
    else:
        print(f"✅ 找到预训练权重: {weights_path}")
    
    encoder, hidden_size = load_model(args.model_name, weights_path, device)
    
    # 创建分类头
    num_classes = len(label_names)
    head = nn.Linear(hidden_size, num_classes).to(device)
    print(f"   分类头: Linear({hidden_size} -> {num_classes})")
    
    # Step 5: 推理
    wait_for_user(5, 5, "运行推理并查看预测结果")
    preds, probs = encode_and_classify(encoder, head, encodings, labels, device)
    print()
    show_predictions(df, labels, preds, probs, label_names)
    
    # 完成
    print()
    print("=" * 70)
    print("🎉 演示完成!")
    print("=" * 70)
    print()
    print("💡 后续建议:")
    print("   1. 使用预训练权重运行: python demo_pipeline.py --use-pretrained")
    print("   2. 使用更多数据: python demo_pipeline.py --n-samples 50")
    print("   3. 查看完整评估脚本: eval_with_encoder_head.py")


if __name__ == "__main__":
    main()
