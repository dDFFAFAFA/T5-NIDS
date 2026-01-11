# %% [markdown]
# # PCAP-Encoder NIDS 微调完整流程 (使用作者组件)
# 
# 本 Notebook 使用作者提供的核心组件进行 NIDS 分类微调：
# - `Classification_Dataset`: 数据加载
# - `Classification_model`: 训练流程
# - `ModelWithBottleneck`: 模型封装
# 
# **使用方法**: 在 VS Code 中使用 "Run Cell" 逐个单元格运行，或在 Jupyter 中打开

# %% [markdown]
# ## 1. 环境配置

# %%
import os
import sys
import glob
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 项目根目录
PROJECT_ROOT = Path("/Users/changye/Desktop/期刊/模块库/Debunk_Traffic_Representation-master/code/PCAP_encoder")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 项目根目录: {PROJECT_ROOT}")
print(f"🖥️ 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"📦 PyTorch: {torch.__version__}")

# %% [markdown]
# ## 2. 导入作者的核心模块

# %%
# 导入作者的核心组件
from Core.classes.dataset_for_classification import Classification_Dataset
from Core.classes.classification_model import Classification_model
from Core.classes.tokenizer import QA_Tokenizer_T5
from Core.classes.logger import TrainingExperimentLogger

print("✅ 作者核心模块导入成功!")
print("   - Classification_Dataset: 数据集类")
print("   - Classification_model: 训练管理类")
print("   - QA_Tokenizer_T5: 分词器")
print("   - TrainingExperimentLogger: 日志记录器")

# %% [markdown]
# ## 3. 配置参数
# 
# 根据需要修改以下配置:

# %%
# ============================================================
# 📋 实验配置 - 根据需要修改这里
# ============================================================

# 数据路径 (绝对路径)
DATA_DIR = PROJECT_ROOT / "data" / "CIC-IDS2017" / "Classification"
TRAINING_DATA = str(DATA_DIR / "train.parquet")
VALIDATION_DATA = str(DATA_DIR / "val.parquet")
TESTING_DATA = str(DATA_DIR / "test.parquet")

# 预训练权重路径
PRETRAINED_MODEL_PATH = str(PROJECT_ROOT / "models" / "pretrained")

# 训练参数
EPOCHS = 20
BATCH_SIZE = 24
LEARNING_RATE = 0.001  # 冻结编码器用 0.001, 解冻用 0.00001
SEED = 42

# 是否冻结编码器
FIX_ENCODER = True  # True = 冻结, False = 解冻微调

# 模型配置
MODEL_NAME = "T5-base"
TOKENIZER_NAME = "T5-base"
BOTTLENECK = "mean"  # mean, first, last, attention
MAX_QST_LENGTH = 512
MAX_ANS_LENGTH = 32

# 其他
LOSS_TYPE = "normal"  # normal 或 weighted (处理类别不平衡)
PERCENTAGE = 100  # 使用数据的百分比 [1, 100]
LOG_LEVEL = "info"
OUTPUT_PATH = str(PROJECT_ROOT / "results")

# 打印配置
print("=" * 60)
print("📋 实验配置")
print("=" * 60)
print(f"📂 训练数据: {TRAINING_DATA}")
print(f"📂 验证数据: {VALIDATION_DATA}")
print(f"📂 测试数据: {TESTING_DATA}")
print(f"📂 预训练权重: {PRETRAINED_MODEL_PATH}")
print(f"🔧 学习率: {LEARNING_RATE}")
print(f"🔧 Epochs: {EPOCHS}")
print(f"🔧 Batch Size: {BATCH_SIZE}")
print(f"🔧 编码器状态: {'冻结' if FIX_ENCODER else '解冻'}")
print(f"🔧 Bottleneck: {BOTTLENECK}")
print("=" * 60)

# %% [markdown]
# ## 4. 数据准备 (可选)
# 
# 如果还没有转换数据，运行此单元格

# %%
def bytes_to_hex(byte_array: np.ndarray, format_type: str = 'every4') -> list:
    """将字节数组转换为十六进制字符串"""
    hex_strings = []
    for row in tqdm(byte_array, desc="转换十六进制", leave=False):
        hex_str = ''.join(f'{int(b):02x}' for b in row)
        if format_type == 'every4':
            hex_str = ' '.join(hex_str[i:i+4] for i in range(0, len(hex_str), 4))
        elif format_type == 'every2':
            hex_str = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
        hex_strings.append(hex_str.strip())
    return hex_strings


def prepare_dataset_if_needed(
    input_dir: Path,
    output_dir: Path,
    max_bytes: int = 64,
    format_type: str = 'every4',
    sample_size: int = None
):
    """如果数据不存在，则准备数据"""
    output_dir = Path(output_dir)
    train_file = output_dir / "train.parquet"
    
    if train_file.exists():
        print(f"✅ 数据已存在: {output_dir}")
        return True
    
    print(f"⏳ 数据不存在，开始转换...")
    input_dir = Path(input_dir)
    
    # 加载原始数据
    parquet_files = sorted(glob.glob(str(input_dir / "Payload_Bytes_File_*.parquet")))
    if not parquet_files:
        print(f"❌ 未找到原始数据文件: {input_dir}")
        return False
    
    print(f"📂 找到 {len(parquet_files)} 个分片")
    dfs = [pd.read_parquet(f) for f in tqdm(parquet_files, desc="加载分片")]
    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ 加载完成: {len(df):,} 样本")
    
    # 获取标签列
    label_col = None
    for col in ['attack_label', 'Label', 'label']:
        if col in df.columns:
            label_col = col
            break
    if not label_col:
        print("❌ 未找到标签列")
        return False
    
    # 采样
    if sample_size and sample_size < len(df):
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, sample_size // df[label_col].nunique())), random_state=SEED)
        )
        print(f"📊 采样后: {len(df):,} 样本")
    
    # 获取 payload 列
    payload_cols = sorted(
        [c for c in df.columns if c.startswith('payload_byte_')],
        key=lambda x: int(x.split('_')[-1])
    )[:max_bytes]
    
    # 转换
    X_bytes = df[payload_cols].values.astype(np.uint8)
    y_labels = df[label_col].values
    contexts = bytes_to_hex(X_bytes, format_type=format_type)
    
    # 标签映射
    unique_labels = sorted(df[label_col].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 构建 DataFrame
    result = pd.DataFrame({
        'question': 'Classify the network packet',
        'context': contexts,
        'class': [label_to_id[label] for label in y_labels],
        'type_q': y_labels
    })
    
    # 划分
    train_df, temp_df = train_test_split(result, test_size=0.4, stratify=result['class'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=SEED)
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_to_id, f, indent=2)
    
    print(f"✅ 数据已保存:")
    print(f"   训练集: {len(train_df):,}")
    print(f"   验证集: {len(val_df):,}")
    print(f"   测试集: {len(test_df):,}")
    return True

# %%
# 检查并准备数据
# 修改 RAW_DATA_DIR 为你的原始 Payload-Bytes 数据目录
RAW_DATA_DIR = PROJECT_ROOT / "data" / "CIC-IDS2017" / "Payload-Bytes"

prepare_dataset_if_needed(
    input_dir=RAW_DATA_DIR,
    output_dir=DATA_DIR,
    max_bytes=64,
    format_type='every4',
    sample_size=10000  # 设为 None 使用全部数据
)

# %% [markdown]
# ## 5. 检查数据格式

# %%
# 检查转换后的数据
if Path(TRAINING_DATA).exists():
    train_df = pd.read_parquet(TRAINING_DATA)
    print(f"📊 训练集大小: {len(train_df):,}")
    print(f"📋 列: {train_df.columns.tolist()}")
    print(f"\n🔍 数据样例:")
    print(train_df.head(3).to_string())
    print(f"\n📊 类别分布:")
    print(train_df['type_q'].value_counts())
else:
    print(f"❌ 训练数据不存在: {TRAINING_DATA}")
    print("   请先运行上一个单元格准备数据")

# %% [markdown]
# ## 6. 构建配置字典 (模拟命令行参数)

# %%
# 构建 opts 字典 (作者的代码使用这个格式)
opts = {
    # 实验标识
    "identifier": f"cicids_notebook_lr{LEARNING_RATE}_{'frozen' if FIX_ENCODER else 'unfrozen'}",
    "experiment": "CIC-IDS2017_NIDS",
    "task": "supervised",
    "clean_start": True,
    
    # 模型配置
    "model_name": MODEL_NAME,
    "tokenizer_name": TOKENIZER_NAME,
    "finetuned_path_model": PRETRAINED_MODEL_PATH,
    "bottleneck": BOTTLENECK,
    "pkt_repr_dim": 768,
    
    # 训练参数
    "lr": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "loss": LOSS_TYPE,
    "fix_encoder": FIX_ENCODER,
    
    # 数据参数
    "training_data": TRAINING_DATA,
    "validation_data": VALIDATION_DATA,
    "testing_data": TESTING_DATA,
    "percentage": PERCENTAGE,
    "max_qst_length": MAX_QST_LENGTH,
    "max_ans_length": MAX_ANS_LENGTH,
    "input_format": "every4",
    
    # 其他
    "output_path": OUTPUT_PATH,
    "log_level": LOG_LEVEL,
    "gpu": "0,",
    "use_cuda": torch.cuda.is_available(),
}

print("✅ 配置字典构建完成!")
for key, value in opts.items():
    print(f"   {key}: {value}")

# %% [markdown]
# ## 7. 初始化日志记录器

# %%
# 初始化日志记录器
print("⏳ 初始化日志记录器...")
logger = TrainingExperimentLogger(opts)
logger.start_experiment(opts)
print(f"✅ 日志记录器初始化完成!")
print(f"   实验ID: {opts['identifier']}")

# %% [markdown]
# ## 8. 初始化分词器

# %%
# 初始化分词器
print("⏳ 初始化分词器...")
tokenizer_obj = QA_Tokenizer_T5(opts)
print(f"✅ 分词器初始化完成!")
print(f"   模型: {opts['tokenizer_name']}")

# %% [markdown]
# ## 9. 加载数据集

# %%
# 加载训练/验证数据集
print("⏳ 加载训练数据集...")
dataset_trainval = Classification_Dataset(opts, tokenizer_obj)
dataset_trainval.load_dataset(
    "Train",
    opts["training_data"],
    opts['input_format'],
    opts["validation_data"],
    opts["percentage"]
)
print(f"✅ 训练/验证数据集加载完成!")
print(f"   训练集大小: {dataset_trainval.size_train}")
print(f"   验证集大小: {dataset_trainval.size_val}")

# %%
# 加载测试数据集
print("⏳ 加载测试数据集...")
dataset_test = Classification_Dataset(opts, tokenizer_obj)
dataset_test.load_dataset("Test", opts["testing_data"], opts['input_format'])
print(f"✅ 测试数据集加载完成!")
print(f"   测试集大小: {len(dataset_test)}")

# %%
# 检查数据样例
print("\n🔍 数据样例检查:")
sample_idx, sample_data = dataset_trainval[0]
print(f"   索引: {sample_idx}")
print(f"   input_ids shape: {sample_data['input_ids'].shape}")
print(f"   attention_mask shape: {sample_data['attention_mask'].shape}")
print(f"   label_class: {sample_data['label_class']}")

# %% [markdown]
# ## 10. 初始化分类模型

# %%
# 初始化分类模型
print("⏳ 初始化分类模型...")
model_obj = Classification_model(opts, tokenizer_obj, dataset_trainval, dataset_test)
print(f"✅ 分类模型初始化完成!")

# %% [markdown]
# ## 11. 开始训练
# 
# ⚠️ 这一步会开始实际训练，可能需要较长时间

# %%
# 开始训练
print("🚀 开始训练...")
print("=" * 60)
model_obj.run(logger, opts)
print("=" * 60)
print("✅ 训练完成!")

# %% [markdown]
# ## 12. 结束实验

# %%
# 结束实验
logger.end_experiment()
print("🎉 实验结束!")

# %% [markdown]
# ## 13. 结果分析 (可选)
# 
# 如果训练完成后想查看结果:

# %%
# 查看结果目录
results_dir = Path(OUTPUT_PATH) / opts['experiment'] / opts['identifier']
if results_dir.exists():
    print(f"📂 结果目录: {results_dir}")
    for f in results_dir.iterdir():
        print(f"   - {f.name}")
else:
    print(f"⚠️ 结果目录不存在: {results_dir}")

# %% [markdown]
# ---
# ## 备注
# 
# ### 如果训练出错，检查以下几点:
# 1. 数据文件是否存在且格式正确
# 2. 预训练权重路径是否正确
# 3. GPU 内存是否足够 (可尝试减小 BATCH_SIZE)
# 4. 查看具体错误信息，定位问题

# %%
# 调试: 手动检查模型定义
print("🔍 调试信息:")
print(f"   CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"   预训练权重路径存在: {Path(PRETRAINED_MODEL_PATH).exists()}")
print(f"   训练数据存在: {Path(TRAINING_DATA).exists()}")
