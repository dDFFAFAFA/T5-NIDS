#!/bin/bash
# =============================================================================
# run_cicids_classification.sh
# =============================================================================
# CIC-IDS2017 数据集分类训练脚本
# 使用 PCAP-Encoder 预训练模型进行 NIDS 微调
# =============================================================================

set -e

# =============================================================================
# 实验配置
# =============================================================================
TASK="supervised"
LOG_LEVEL="info"
OUTPUT_PATH="./results/"
EXPERIMENT="CIC-IDS2017_NIDS"

# =============================================================================
# 模型配置
# =============================================================================
# 预训练权重路径（相对于 Experiments 目录）
FINETUNED_PATH_MODEL="../models/pretrained"
MODEL_NAME="T5-base"
TOKENIZER_NAME="T5-base"

# Bottleneck 设置: mean, first, last, attention
BOTTLENECK="mean"
PKT_REPR_DIM=768

# =============================================================================
# GPU 配置
# =============================================================================
GPU=(0)
GPU_STRING="$(IFS=, ; echo "${GPU[*]}"),"
PORT=29500
export GPUS_PER_NODE=1

# =============================================================================
# 训练参数
# =============================================================================
BATCH_SIZE=24
EPOCHS=20
MAX_QST_LENGTH=512
MAX_ANS_LENGTH=32
PERC=100  # 使用数据的百分比 [1, 100]
SEED=42
LOSS="normal"  # normal 或 weighted (处理类别不平衡)

# 学习率设置
# - 冻结编码器: 0.001
# - 解冻编码器: 0.00001
LR=0.001

# 是否冻结编码器
# - 设置为 true: 添加 --fix_encoder 参数
# - 设置为 false: 解冻编码器，建议降低 LR
FIX_ENCODER=true

# =============================================================================
# 数据路径 (相对于 Experiments 目录)
# =============================================================================
DATA_DIR="../data/CIC-IDS2017/Classification"
TRAINING_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/val.parquet"
TEST_DATA="${DATA_DIR}/test.parquet"

# =============================================================================
# 实验标识符
# =============================================================================
if [ "$FIX_ENCODER" = true ]; then
    ENCODER_STATUS="frozen"
else
    ENCODER_STATUS="unfrozen"
fi
IDENTIFIER="cicids_lr${LR}_seed${SEED}_loss${LOSS}_batch${BATCH_SIZE}_${ENCODER_STATUS}"

# =============================================================================
# 构建运行参数
# =============================================================================
export SCRIPT=../2.Training/classification/classification.py

SCRIPT_ARGS=" \
    --identifier $IDENTIFIER \
    --experiment $EXPERIMENT \
    --task $TASK \
    --clean_start \
    --tokenizer_name $TOKENIZER_NAME \
    --lr $LR \
    --loss $LOSS \
    --model_name $MODEL_NAME \
    --log_level $LOG_LEVEL \
    --output_path $OUTPUT_PATH \
    --training_data $TRAINING_DATA \
    --validation_data $VAL_DATA \
    --testing_data $TEST_DATA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --bottleneck $BOTTLENECK \
    --max_qst_length $MAX_QST_LENGTH \
    --max_ans_length $MAX_ANS_LENGTH \
    --percentage $PERC \
    --gpu $GPU_STRING \
    --finetuned_path_model $FINETUNED_PATH_MODEL \
"

# 添加冻结编码器参数
if [ "$FIX_ENCODER" = true ]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --fix_encoder"
fi

# =============================================================================
# 运行训练
# =============================================================================
echo "============================================================="
echo "🚀 CIC-IDS2017 NIDS 分类训练"
echo "============================================================="
echo "📋 实验标识符: $IDENTIFIER"
echo "📂 训练数据: $TRAINING_DATA"
echo "📂 验证数据: $VAL_DATA"
echo "📂 测试数据: $TEST_DATA"
echo "🔧 学习率: $LR"
echo "🔧 Batch Size: $BATCH_SIZE"
echo "🔧 Epochs: $EPOCHS"
echo "🔧 编码器状态: $ENCODER_STATUS"
echo "🔧 Bottleneck: $BOTTLENECK"
echo "============================================================="

# 检查数据文件是否存在
if [ ! -f "$TRAINING_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAINING_DATA"
    echo "   请先运行 prepare_cicids_dataset.py 生成数据"
    exit 1
fi

# 运行训练
accelerate launch \
    --num_processes=$GPUS_PER_NODE \
    --main_process_port=$PORT \
    $SCRIPT $SCRIPT_ARGS

echo "✅ 训练完成!"
