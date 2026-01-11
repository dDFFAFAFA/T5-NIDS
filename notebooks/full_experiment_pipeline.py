#!/usr/bin/env python
# coding: utf-8

# # PCAP_encoder 完整实验流程
# 
# 本notebook演示PCAP_encoder的完整实验链路：
# 1. 数据加载与合并（18个分片）
# 2. 数据预处理与可视化
# 3. 加载预训练T5模型
# 4. 特征提取与编码
# 5. 分类器训练与评估
# 6. 结果可视化

# ## 1. 环境配置与依赖导入

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}")
print(f"📦 PyTorch 版本: {torch.__version__}")

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("✅ 环境配置完成!")

# ## 2. 数据加载与合并

# 数据路径
DATA_DIR = Path("/home/test/ybk/nids/encoder/PCAP_encoder/data/CIC-IDS2017/Payload-Bytes")
WEIGHTS_PATH = Path("../models/weights.pth")

# 查找所有分片文件
parquet_files = sorted(glob.glob(str(DATA_DIR / "Payload_Bytes_File_*.parquet")))
print(f"📂 找到 {len(parquet_files)} 个数据分片:")
for f in parquet_files[:5]:
    print(f"   - {Path(f).name}")
if len(parquet_files) > 5:
    print(f"   ... 还有 {len(parquet_files) - 5} 个文件")

# 加载并合并所有分片
print("\n⏳ 加载数据中...")
dfs = []
for f in tqdm(parquet_files, desc="加载分片"):
    df_part = pd.read_parquet(f)
    dfs.append(df_part)

df_full = pd.concat(dfs, ignore_index=True)
print(f"\n✅ 数据加载完成!")
print(f"   总样本数: {len(df_full):,}")
print(f"   列数: {len(df_full.columns)}")
print(f"   内存占用: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ## 3. 数据探索与可视化

# 查看数据结构
print("📋 数据列信息:")
print(df_full.columns.tolist()[:20])
print(f"... 共 {len(df_full.columns)} 列")

# 获取标签列
if 'attack_label' in df_full.columns:
    label_col = 'attack_label'
elif 'Label' in df_full.columns:
    label_col = 'Label'
else:
    # 尝试找到标签列
    for col in df_full.columns:
        if 'label' in col.lower() or 'attack' in col.lower():
            label_col = col
            break
    else:
        label_col = df_full.columns[-1]  # 使用最后一列

print(f"\n🏷️ 标签列: {label_col}")

# 标签分布
label_counts = df_full[label_col].value_counts()
print(f"\n📊 标签分布:")
for label, count in label_counts.items():
    pct = count / len(df_full) * 100
    print(f"   {label}: {count:,} ({pct:.2f}%)")

# 可视化标签分布
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 柱状图
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 0.8, len(label_counts)))
bars = ax1.bar(range(len(label_counts)), label_counts.values, color=colors)
ax1.set_xticks(range(len(label_counts)))
ax1.set_xticklabels(label_counts.index, rotation=45, ha='right')
ax1.set_xlabel('攻击类型')
ax1.set_ylabel('样本数量')
ax1.set_title('各类别样本数量分布')
for bar, count in zip(bars, label_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'{count:,}', ha='center', va='bottom', fontsize=10)

# 饼图
ax2 = axes[1]
ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('各类别样本比例')

plt.tight_layout()
plt.savefig('../docs/label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 标签分布图已保存!")

# ## 4. 数据采样与预处理

# 由于数据量可能很大，进行采样
SAMPLE_SIZE = 10000  # 采样数量，可根据内存调整
MAX_BYTES = 64  # 使用的最大字节数

print(f"\n🔄 数据采样...")
print(f"   采样大小: {SAMPLE_SIZE:,}")
print(f"   使用字节数: {MAX_BYTES}")

# 分层采样
df_sampled = df_full.groupby(label_col, group_keys=False).apply(
    lambda x: x.sample(min(len(x), SAMPLE_SIZE // len(label_counts)), random_state=SEED)
)
print(f"   采样后样本数: {len(df_sampled):,}")

# 采样后的标签分布
sampled_counts = df_sampled[label_col].value_counts()
print(f"\n📊 采样后标签分布:")
for label, count in sampled_counts.items():
    pct = count / len(df_sampled) * 100
    print(f"   {label}: {count:,} ({pct:.2f}%)")

# 获取 payload 列
payload_cols = sorted(
    [c for c in df_sampled.columns if c.startswith('payload_byte_')],
    key=lambda x: int(x.split('_')[-1])
)[:MAX_BYTES]

print(f"\n📦 Payload 列数: {len(payload_cols)}")

# 提取字节数据和标签
X_bytes = df_sampled[payload_cols].values.astype(np.uint8)
y_labels = df_sampled[label_col].values

print(f"   字节数据形状: {X_bytes.shape}")
print(f"   标签数据形状: {y_labels.shape}")

# 可视化部分字节数据
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# 不同类别的字节模式
unique_labels = np.unique(y_labels)[:3]  # 取前3个类别
for i, label in enumerate(unique_labels):
    ax = axes[i]
    idx = np.where(y_labels == label)[0][:5]  # 每个类别取5个样本
    for j, sample_idx in enumerate(idx):
        ax.plot(X_bytes[sample_idx], alpha=0.7, label=f'样本 {j+1}')
    ax.set_title(f'类别: {label}')
    ax.set_xlabel('字节位置')
    ax.set_ylabel('字节值')
    ax.legend(loc='upper right')
    ax.set_xlim(0, MAX_BYTES)

plt.tight_layout()
plt.savefig('../docs/byte_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 字节模式图已保存!")

# ## 5. 转换为十六进制格式

def bytes_to_hex(byte_array: np.ndarray, format_type: str = 'every4') -> List[str]:
    """将字节数组转换为十六进制字符串"""
    hex_strings = []
    for row in tqdm(byte_array, desc="转换十六进制"):
        hex_str = ''.join(f'{int(b):02x}' for b in row)
        if format_type == 'every4':
            hex_str = ' '.join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
        elif format_type == 'every2':
            hex_str = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
        hex_strings.append(hex_str)
    return hex_strings

print("\n🔄 转换为十六进制格式...")
contexts = bytes_to_hex(X_bytes, format_type='every4')

print(f"\n📝 转换结果示例:")
for i in range(3):
    print(f"   样本 {i+1} ({y_labels[i]}):")
    print(f"   原始字节: {X_bytes[i][:8].tolist()}")
    print(f"   十六进制: {contexts[i][:50]}...")
    print()

# ## 6. 标签编码

# 创建标签映射
unique_labels_all = np.unique(y_labels)
label_to_id = {label: idx for idx, label in enumerate(unique_labels_all)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

y_encoded = np.array([label_to_id[label] for label in y_labels])
num_classes = len(unique_labels_all)

print(f"🏷️ 标签编码:")
for label, idx in label_to_id.items():
    print(f"   {label} -> {idx}")
print(f"\n   类别总数: {num_classes}")

# ## 7. 加载 T5 模型和预训练权重

from transformers import T5ForConditionalGeneration, T5TokenizerFast

MODEL_NAME = "t5-base"

print(f"\n⏳ 加载 T5 模型: {MODEL_NAME}...")
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# 加载预训练权重
if WEIGHTS_PATH.exists():
    print(f"📥 加载预训练权重: {WEIGHTS_PATH}")
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("✅ 预训练权重加载成功!")
    except Exception as e:
        print(f"⚠️ 权重加载失败: {e}")
        print("   将使用随机初始化权重")
else:
    print(f"⚠️ 未找到预训练权重: {WEIGHTS_PATH}")
    print("   将使用随机初始化权重")

# 提取编码器并冻结
encoder = model.encoder.to(device)
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()

hidden_size = model.config.d_model
print(f"\n📊 模型信息:")
print(f"   编码器隐藏维度: {hidden_size}")
print(f"   编码器层数: {model.config.num_layers}")
print(f"   注意力头数: {model.config.num_heads}")

# ## 8. 构建问答格式输入并分词

QUESTION = "Classify the network packet"
MAX_LENGTH = 512

print(f"\n🔤 构建模型输入...")
print(f"   问题模板: '{QUESTION}'")
print(f"   最大长度: {MAX_LENGTH}")

# 构建输入文本
input_texts = [f"question: {QUESTION} context: {ctx}" for ctx in contexts]

print(f"\n📝 输入示例:")
print(f"   {input_texts[0][:100]}...")

# 分词
print(f"\n⏳ 分词编码中...")
encodings = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

print(f"✅ 分词完成!")
print(f"   input_ids 形状: {encodings['input_ids'].shape}")
print(f"   attention_mask 形状: {encodings['attention_mask'].shape}")

# 展示分词结果
print(f"\n📝 分词结果示例:")
sample_tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][0][:30])
print(f"   Tokens (前30): {sample_tokens}")

# ## 9. 数据集划分

print(f"\n🔄 划分数据集...")
X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
    np.arange(len(y_encoded)), y_encoded,
    test_size=0.4, stratify=y_encoded, random_state=SEED
)

X_val_idx, X_test_idx, y_val, y_test = train_test_split(
    X_temp_idx, y_temp,
    test_size=0.5, stratify=y_temp, random_state=SEED
)

print(f"   训练集: {len(X_train_idx):,} 样本 ({len(X_train_idx)/len(y_encoded)*100:.1f}%)")
print(f"   验证集: {len(X_val_idx):,} 样本 ({len(X_val_idx)/len(y_encoded)*100:.1f}%)")
print(f"   测试集: {len(X_test_idx):,} 样本 ({len(X_test_idx)/len(y_encoded)*100:.1f}%)")

# 创建数据加载器
BATCH_SIZE = 32

def create_loader(indices, shuffle=False):
    input_ids = encodings['input_ids'][indices]
    attention_mask = encodings['attention_mask'][indices]
    labels = torch.tensor(y_encoded[indices], dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = create_loader(X_train_idx, shuffle=True)
val_loader = create_loader(X_val_idx, shuffle=False)
test_loader = create_loader(X_test_idx, shuffle=False)

print(f"\n📦 DataLoader 创建完成:")
print(f"   训练批次数: {len(train_loader)}")
print(f"   验证批次数: {len(val_loader)}")
print(f"   测试批次数: {len(test_loader)}")

# ## 10. 特征提取

@torch.no_grad()
def extract_features(loader, encoder, device, bottleneck='mean'):
    """使用编码器提取特征"""
    encoder.eval()
    all_features = []
    all_labels = []
    
    for batch in tqdm(loader, desc="提取特征"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2]
        
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        
        if bottleneck == 'mean':
            features = hidden_states.mean(dim=1)
        elif bottleneck == 'first':
            features = hidden_states[:, 0, :]
        elif bottleneck == 'last':
            features = hidden_states[:, -1, :]
        else:
            features = hidden_states.mean(dim=1)
        
        all_features.append(features.cpu())
        all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)

print("\n⏳ 提取特征中...")
print("   使用瓶颈层: mean pooling")

train_features, train_labels = extract_features(train_loader, encoder, device)
val_features, val_labels = extract_features(val_loader, encoder, device)
test_features, test_labels = extract_features(test_loader, encoder, device)

print(f"\n✅ 特征提取完成!")
print(f"   训练集特征: {train_features.shape}")
print(f"   验证集特征: {val_features.shape}")
print(f"   测试集特征: {test_features.shape}")

# 可视化特征分布 (使用 t-SNE 或 PCA)
from sklearn.decomposition import PCA

print("\n🎨 可视化特征分布 (PCA)...")
pca = PCA(n_components=2, random_state=SEED)
test_features_2d = pca.fit_transform(test_features.numpy())

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(
    test_features_2d[:, 0], test_features_2d[:, 1],
    c=test_labels.numpy(), cmap='viridis',
    alpha=0.6, s=30
)
plt.colorbar(scatter, label='类别')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('T5 编码器特征 (PCA 降维)')

# 添加类别标注
for i, label in enumerate(unique_labels_all[:5]):  # 只标注前5个类别
    idx = np.where(test_labels.numpy() == i)[0]
    if len(idx) > 0:
        center = test_features_2d[idx].mean(axis=0)
        ax.annotate(label, center, fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('../docs/feature_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 特征分布图已保存!")

# ## 11. 训练分类器

print("\n🏋️ 训练分类器...")

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

classifier = Classifier(hidden_size, num_classes).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练参数
EPOCHS = 30

# 移动特征到设备
train_features = train_features.to(device)
train_labels = train_labels.to(device)
val_features = val_features.to(device)
val_labels = val_labels.to(device)

# 训练循环
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0
best_epoch = 0

for epoch in range(EPOCHS):
    # 训练
    classifier.train()
    optimizer.zero_grad()
    
    logits = classifier(train_features)
    train_loss = criterion(logits, train_labels)
    train_loss.backward()
    optimizer.step()
    
    train_preds = logits.argmax(dim=1)
    train_acc = (train_preds == train_labels).float().mean().item()
    
    # 验证
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_features)
        val_loss = criterion(val_logits, val_labels)
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == val_labels).float().mean().item()
    
    # 记录历史
    history['train_loss'].append(train_loss.item())
    history['val_loss'].append(val_loss.item())
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(classifier.state_dict(), '../models/best_classifier.pth')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
              f"Train Loss={train_loss.item():.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss.item():.4f}, Acc={val_acc:.4f}")

print(f"\n✅ 训练完成!")
print(f"   最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch + 1})")

# 可视化训练过程
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲线
ax1 = axes[0]
ax1.plot(history['train_loss'], label='训练损失', linewidth=2)
ax1.plot(history['val_loss'], label='验证损失', linewidth=2)
ax1.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'最佳模型 (Epoch {best_epoch+1})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练和验证损失')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy 曲线
ax2 = axes[1]
ax2.plot(history['train_acc'], label='训练准确率', linewidth=2)
ax2.plot(history['val_acc'], label='验证准确率', linewidth=2)
ax2.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'最佳模型 (Epoch {best_epoch+1})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('训练和验证准确率')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 训练曲线图已保存!")

# ## 12. 测试评估

print("\n📊 测试集评估...")

# 加载最佳模型
classifier.load_state_dict(torch.load('../models/best_classifier.pth', weights_only=True))
classifier.eval()

# 测试
test_features = test_features.to(device)
test_labels = test_labels.to(device)

with torch.no_grad():
    test_logits = classifier(test_features)
    test_probs = F.softmax(test_logits, dim=1)
    test_preds = test_logits.argmax(dim=1)

test_preds_np = test_preds.cpu().numpy()
test_labels_np = test_labels.cpu().numpy()
test_probs_np = test_probs.cpu().numpy()

# 计算指标
accuracy = accuracy_score(test_labels_np, test_preds_np)
f1_macro = f1_score(test_labels_np, test_preds_np, average='macro')
f1_weighted = f1_score(test_labels_np, test_preds_np, average='weighted')
precision = precision_score(test_labels_np, test_preds_np, average='macro')
recall = recall_score(test_labels_np, test_preds_np, average='macro')

print(f"\n📈 测试集指标:")
print(f"   准确率 (Accuracy):     {accuracy*100:.2f}%")
print(f"   F1 Score (Macro):      {f1_macro*100:.2f}%")
print(f"   F1 Score (Weighted):   {f1_weighted*100:.2f}%")
print(f"   精确率 (Precision):    {precision*100:.2f}%")
print(f"   召回率 (Recall):       {recall*100:.2f}%")

# 分类报告
print(f"\n📋 详细分类报告:")
target_names = [id_to_label[i] for i in range(num_classes)]
print(classification_report(test_labels_np, test_preds_np, target_names=target_names))

# ## 13. 可视化分类结果

# 混淆矩阵
print("\n🎨 绘制混淆矩阵...")
cm = confusion_matrix(test_labels_np, test_preds_np)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 原始混淆矩阵
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=target_names, yticklabels=target_names)
ax1.set_xlabel('预测标签')
ax1.set_ylabel('真实标签')
ax1.set_title('混淆矩阵 (数量)')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)

# 归一化混淆矩阵
ax2 = axes[1]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
            xticklabels=target_names, yticklabels=target_names)
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')
ax2.set_title('混淆矩阵 (归一化)')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('../docs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 混淆矩阵图已保存!")

# ## 14. ROC 曲线

print("\n🎨 绘制 ROC 曲线...")

# 二值化标签
y_test_bin = label_binarize(test_labels_np, classes=range(num_classes))

# 计算每个类别的 ROC 曲线
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均 ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_probs_np.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 绘制 ROC 曲线
fig, ax = plt.subplots(figsize=(12, 10))

# 绘制微平均 ROC
ax.plot(fpr["micro"], tpr["micro"],
        label=f'微平均 ROC (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=3)

# 绘制每个类别的 ROC
colors = plt.cm.viridis(np.linspace(0, 0.8, num_classes))
for i, color in enumerate(colors):
    if i < len(target_names):
        ax.plot(fpr[i], tpr[i], color=color, linewidth=2,
                label=f'{target_names[i]} (AUC = {roc_auc[i]:.3f})')

# 绘制对角线
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假阳率 (False Positive Rate)')
ax.set_ylabel('真阳率 (True Positive Rate)')
ax.set_title('多分类 ROC 曲线')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ ROC 曲线图已保存!")

# ## 15. 每类别性能对比

print("\n🎨 绘制各类别性能对比...")

# 计算每个类别的指标
per_class_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for i in range(num_classes):
    # 计算每个类别的指标
    class_mask = test_labels_np == i
    if class_mask.sum() > 0:
        class_acc = (test_preds_np[class_mask] == i).mean()
        per_class_metrics['accuracy'].append(class_acc)
    else:
        per_class_metrics['accuracy'].append(0)

# 使用 sklearn 计算 per-class 指标
per_class_precision = precision_score(test_labels_np, test_preds_np, average=None, zero_division=0)
per_class_recall = recall_score(test_labels_np, test_preds_np, average=None, zero_division=0)
per_class_f1 = f1_score(test_labels_np, test_preds_np, average=None, zero_division=0)

per_class_metrics['precision'] = per_class_precision.tolist()
per_class_metrics['recall'] = per_class_recall.tolist()
per_class_metrics['f1'] = per_class_f1.tolist()

# 绘制性能对比图
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(num_classes)
width = 0.2

bars1 = ax.bar(x - 1.5*width, per_class_metrics['accuracy'], width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x - 0.5*width, per_class_metrics['precision'], width, label='Precision', color='coral')
bars3 = ax.bar(x + 0.5*width, per_class_metrics['recall'], width, label='Recall', color='seagreen')
bars4 = ax.bar(x + 1.5*width, per_class_metrics['f1'], width, label='F1-Score', color='orchid')

ax.set_xlabel('类别')
ax.set_ylabel('分数')
ax.set_title('各类别性能指标对比')
ax.set_xticks(x)
ax.set_xticklabels(target_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标注
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.1:
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../docs/per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 各类别性能图已保存!")

# ## 16. 预测结果展示

print("\n📋 预测结果示例:")
print("-" * 80)

# 随机选择 10 个样本展示
np.random.seed(SEED)
sample_indices = np.random.choice(len(test_labels_np), size=10, replace=False)

correct_count = 0
for idx in sample_indices:
    true_label = target_names[test_labels_np[idx]]
    pred_label = target_names[test_preds_np[idx]]
    confidence = test_probs_np[idx].max() * 100
    is_correct = "✅" if true_label == pred_label else "❌"
    if true_label == pred_label:
        correct_count += 1
    
    print(f"样本 {idx:4d}: 真实={true_label:15s} | 预测={pred_label:15s} | 置信度={confidence:5.1f}% | {is_correct}")

print("-" * 80)
print(f"示例准确率: {correct_count}/10 = {correct_count/10*100:.1f}%")

# ## 17. 实验总结

print("\n" + "="*80)
print("📊 实验总结")
print("="*80)

print(f"""
🔬 实验配置:
   - 数据集: CIC-IDS2017 Payload-Bytes
   - 数据分片: {len(parquet_files)} 个文件
   - 采样大小: {SAMPLE_SIZE:,}
   - Payload 字节数: {MAX_BYTES}
   - 类别数: {num_classes}

🧠 模型配置:
   - 编码器: {MODEL_NAME}
   - 隐藏维度: {hidden_size}
   - 瓶颈层: mean pooling
   - 分类器: 2层 MLP ({hidden_size} -> 256 -> {num_classes})

📈 最终性能:
   - 测试准确率: {accuracy*100:.2f}%
   - F1 Score (Macro): {f1_macro*100:.2f}%
   - F1 Score (Weighted): {f1_weighted*100:.2f}%
   - 精确率: {precision*100:.2f}%
   - 召回率: {recall*100:.2f}%

💾 保存的文件:
   - 分类器权重: ../models/best_classifier.pth
   - 标签分布图: ../docs/label_distribution.png
   - 字节模式图: ../docs/byte_patterns.png
   - 特征可视化: ../docs/feature_visualization.png
   - 训练曲线图: ../docs/training_curves.png
   - 混淆矩阵图: ../docs/confusion_matrix.png
   - ROC 曲线图: ../docs/roc_curves.png
   - 类别性能图: ../docs/per_class_metrics.png
""")

print("🎉 实验完成!")
