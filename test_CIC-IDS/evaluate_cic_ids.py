"""
此脚本是评估 CIC-IDS2017 数据集的初始模板。
它展示了如何使用 nids-datasets 加载数据，并使用预训练模型进行分类评估。
注意：此脚本中的路径和部分逻辑可能需要根据实际环境进行调整。
"""

import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nids_datasets import Dataset

# 1. 加载数据
def load_data(parquet_file):
    """
    从指定的 Parquet 文件中读取数据。
    """
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_file)
    return df

# 2. 准备数据
def prepare_data(df):
    """
    将原始字节数据转换为模型可用的文本格式，并处理标签。
    """
    # 将 payload 字节列（假设有 1500 列）拼接成十六进制字符串作为 context
    # 注意：实际数据集中列名可能从 payload_byte_1 开始，且数量可能不足 1500
    payload_cols = [f'payload_byte_{i}' for i in range(1, 1501) if f'payload_byte_{i}' in df.columns]
    context = df[payload_cols].applymap(lambda x: format(int(x), '02x')).agg(''.join, axis=1)
    
    # 生成问题（Question），作为 T5 等模型的输入前缀
    question = ['Classify the network packet'] * len(df)

    # 标签映射：将字符串标签（如 'BENIGN', 'FTP-Patator'）映射为整数 ID
    label_mapping = {label: idx for idx, label in enumerate(df['attack_label'].unique())}
    df['class'] = df['attack_label'].map(label_mapping)
    
    return context, question, df['class'], df['attack_label']

# 3. 加载预训练模型
def load_model(model_path):
    """
    加载 HuggingFace 格式的预训练模型和分词器。
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# 4. 评估模型
def evaluate_model(model, tokenizer, context, question, labels):
    """
    在给定数据上运行模型并计算 F1 和 AUC 指标。
    """
    # 对 (Question, Context) 对进行分词
    inputs = tokenizer(list(zip(question, context)), padding=True, truncation=True, return_tensors="pt")
    
    # 模型前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取预测类别（Logits 中最大值对应的索引）
    predictions = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
    
    # 计算 Macro F1-score
    f1 = f1_score(labels, predictions, average='macro')
    
    # 计算多分类 AUC (One-vs-Rest)
    # 注意：如果数据集中只有一类，此步骤会报错
    try:
        auc = roc_auc_score(labels, outputs.logits.detach().cpu().numpy(), multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    
    return f1, auc

# 主函数
def main():
    # 数据路径（示例路径，需根据实际情况修改）
    parquet_file = 'CIC-IDS2017/Network-Flows+Packet-Fields+Payload-Bytes/Network_Flows+Packet_Fields+Payload_Bytes_File_1.parquet'
    
    # 加载数据
    try:
        df = load_data(parquet_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {parquet_file}")
        return
    
    # 处理数据
    context, question, labels, _ = prepare_data(df)
    
    # 加载预训练模型
    model_path = 'models/weights.pth'  # 预训练模型权重路径
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 评估模型
    f1, auc = evaluate_model(model, tokenizer, context, question, labels)
    
    # 输出结果
    print(f'F1 Macro: {f1:.4f}')
    print(f'AUC: {auc:.4f}')

if __name__ == '__main__':
    main()

