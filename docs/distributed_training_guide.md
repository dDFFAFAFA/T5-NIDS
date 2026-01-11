# 🔥 单卡 vs 多卡分布式训练：完整指南

> 本文档详细对比单卡训练与多卡分布式训练的核心区别，并深入讲解 DataLoader 和 Accelerate 的底层原理。

---

## 📊 对比总览

| 方面 | 单卡训练 | 多卡分布式训练 |
|------|---------|---------------|
| GPU 指定 | `device = torch.device("cuda:0")` | `os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"` |
| 模型放置 | `model.to(device)` | `accelerator.prepare(model)` |
| 数据加载 | 普通 DataLoader | Accelerate 自动分片 |
| 梯度更新 | 直接 `loss.backward()` | 自动跨卡同步 |
| Batch Size | 即实际 batch size | 每卡 batch × 卡数 = 总 batch |
| 启动方式 | 直接运行 | `notebook_launcher` 或 `accelerate launch` |

---

## 🖥️ 方式一：单卡训练

### 核心代码框架

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ============ 1. 指定设备 ============
device = torch.device("cuda:0")  # 指定第 0 张卡
# 或者: device = torch.device("cuda:2")  # 指定第 2 张卡

# ============ 2. 模型准备 ============
model = MyModel()
model.to(device)  # 模型移到 GPU

# ============ 3. 数据加载 ============
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ============ 4. 优化器 ============
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ============ 5. 训练循环 ============
for epoch in range(num_epochs):
    for batch in train_loader:
        # 数据移到 GPU
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item()}")

# ============ 6. 保存模型 ============
torch.save(model.state_dict(), "model.pth")
```

### 单卡优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 代码简单直观 | ❌ 显存受限 |
| ✅ 调试方便 | ❌ 训练速度慢 |
| ✅ 无通信开销 | ❌ 无法利用多卡资源 |

---

## 🚀 方式二：多卡分布式训练 (使用 Accelerate)

### 核心代码框架

```python
import os
# ⚠️ 必须在 import torch 之前设置！
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import torch
from accelerate import Accelerator, notebook_launcher
from torch.utils.data import DataLoader

def train_distributed():
    """
    🔥 每个 GPU 进程都会独立执行这个函数
    Accelerate 会自动：
    - 分配不同的 GPU 给每个进程
    - 分片数据，每个进程处理 1/N 的数据
    - 同步梯度
    """
    
    # ============ 1. 创建 Accelerator ============
    accelerator = Accelerator()
    # accelerator.device 会自动分配当前进程的 GPU
    
    # ============ 2. 模型准备 (不需要手动 .to(device)) ============
    model = MyModel()
    
    # ============ 3. 数据加载 ============
    train_loader = DataLoader(dataset, batch_size=16)  # 每卡 16
    # 总 batch = 16 × 4卡 = 64
    
    # ============ 4. 优化器 ============
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # ============ 5. 🔥 Accelerate 包装 (核心！) ============
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    # ↑ 这一步会：
    #   - 自动将模型移到当前 GPU
    #   - 用 DistributedDataParallel 包装模型
    #   - 用 DistributedSampler 分片数据
    
    # ============ 6. 训练循环 ============
    for epoch in range(num_epochs):
        for batch in train_loader:
            # ⚠️ 不需要手动 .to(device)！
            inputs = batch['input_ids']
            labels = batch['labels']
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            accelerator.backward(loss)  # 🔥 使用 accelerator.backward()
            optimizer.step()
            
            # 只在主进程打印
            if accelerator.is_main_process:
                print(f"Loss: {loss.item()}")
    
    # ============ 7. 保存模型 (只在主进程) ============
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "model.pth")

# ============ 8. 🔥 启动分布式训练 ============
# Notebook 中：
notebook_launcher(train_distributed, num_processes=4, use_port="29501")

# 命令行中：
# accelerate launch --num_processes=4 train.py
```

---

## 🔑 核心区别总结

### 1. GPU 指定方式

```python
# 单卡
device = torch.device("cuda:2")
model.to(device)
data.to(device)

# 多卡
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # 在 import torch 之前
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(...)  # 自动分配
```

### 2. 反向传播

```python
# 单卡
loss.backward()

# 多卡
accelerator.backward(loss)  # 自动同步梯度
```

### 3. 数据处理

```python
# 单卡
for batch in dataloader:
    inputs = batch['x'].to(device)  # 手动移动

# 多卡
# accelerator.prepare() 后，DataLoader 自动分片
# 不需要手动 .to(device)
for batch in dataloader:
    inputs = batch['x']  # 已经在正确的 GPU 上了
```

### 4. 打印与保存

```python
# 单卡
print(f"Loss: {loss.item()}")
torch.save(model.state_dict(), "model.pth")

# 多卡 (避免重复打印/保存)
if accelerator.is_main_process:
    print(f"Loss: {loss.item()}")
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), "model.pth")
```

### 5. Batch Size 计算

```python
# 单卡
batch_size = 64  # 实际就是 64

# 多卡 (4 张卡)
batch_size = 16  # 每卡 16
# 总 batch = 16 × 4 = 64
# 学习率可能需要 × 4 (线性缩放)
```

---

## 📋 一张图总结

```
┌─────────────────────────────────────────────────────────────────┐
│                        单卡训练                                   │
├─────────────────────────────────────────────────────────────────┤
│  device = torch.device("cuda:0")                                │
│  model.to(device)                                               │
│  data.to(device)                                                │
│  loss.backward()                                                │
│  torch.save(model.state_dict(), ...)                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        多卡分布式训练                             │
├─────────────────────────────────────────────────────────────────┤
│  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # import前!     │
│  accelerator = Accelerator()                                    │
│  model, opt, loader = accelerator.prepare(model, opt, loader)   │
│  accelerator.backward(loss)                                     │
│  if accelerator.is_main_process:                                │
│      torch.save(accelerator.unwrap_model(model).state_dict())   │
│                                                                 │
│  启动: notebook_launcher(fn, num_processes=4)                   │
│     或: accelerate launch --num_processes=4 script.py           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ 常见陷阱

| 陷阱 | 原因 | 解决方案 |
|------|------|---------|
| `CUDA_VISIBLE_DEVICES` 不生效 | 在 `import torch` 之后设置 | 必须在最开头设置 |
| 打印输出重复 N 次 | N 个进程都在打印 | 用 `if accelerator.is_main_process:` |
| 保存的模型无法加载 | 保存了 DDP 包装后的模型 | 用 `accelerator.unwrap_model()` |
| `notebook_launcher` 报错 | 之前已创建 Accelerator | 重启 Kernel |
| 显存不均匀 | 某些操作只在主进程 | 确保所有进程执行相同代码 |

---

# 📚 深入理解：DataLoader 底层原理

## DataLoader 的核心组件

```
DataLoader
    │
    ├── Dataset         # 数据源，实现 __getitem__ 和 __len__
    │
    ├── Sampler         # 决定取数据的顺序
    │   ├── SequentialSampler     # 顺序采样 [0, 1, 2, 3, ...]
    │   ├── RandomSampler         # 随机打乱
    │   └── DistributedSampler    # 🔥 分布式分片采样
    │
    ├── BatchSampler    # 将多个索引组成 batch
    │
    └── collate_fn      # 将多个样本合并为一个 batch
```

## 1. Dataset

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)  # 数据集大小
    
    def __getitem__(self, idx):
        # 返回第 idx 个样本
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }
```

## 2. Sampler（采样器）

### 单卡采样

```python
# shuffle=True 时使用 RandomSampler
# 假设数据集有 10 个样本 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# RandomSampler 会生成随机顺序: [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]

loader = DataLoader(dataset, batch_size=3, shuffle=True)
# Batch 1: [7, 2, 9]
# Batch 2: [0, 5, 1]
# Batch 3: [8, 3, 6]
# Batch 4: [4]  (剩余的)
```

### 多卡分布式采样 (DistributedSampler)

```python
# 假设数据集有 12 个样本，使用 4 张 GPU
# DistributedSampler 会将数据分成 4 份：

# GPU 0 (rank=0): [0, 4, 8]    # 每隔 4 个取一个
# GPU 1 (rank=1): [1, 5, 9]
# GPU 2 (rank=2): [2, 6, 10]
# GPU 3 (rank=3): [3, 7, 11]

# 每个 GPU 只看到 1/4 的数据！
```

### DistributedSampler 源码核心逻辑

```python
class DistributedSampler:
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas  # GPU 数量
        self.rank = rank                  # 当前 GPU 编号
        
        # 计算每个 GPU 处理的样本数
        self.num_samples = len(dataset) // num_replicas
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        # 打乱顺序（所有 GPU 使用相同的种子保证一致）
        random.seed(self.epoch)
        random.shuffle(indices)
        
        # 🔥 关键：每个 GPU 只取属于自己的那部分
        # rank=0 取 [0, 4, 8, ...]
        # rank=1 取 [1, 5, 9, ...]
        indices = indices[self.rank::self.num_replicas]
        
        return iter(indices)
```

## 3. collate_fn（合并函数）

```python
# 默认的 collate_fn 会将多个样本堆叠成 batch
# 输入: [{'x': tensor([1,2])}, {'x': tensor([3,4])}, {'x': tensor([5,6])}]
# 输出: {'x': tensor([[1,2], [3,4], [5,6]])}  # shape: (3, 2)

# 自定义 collate_fn 示例
def my_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

loader = DataLoader(dataset, batch_size=32, collate_fn=my_collate)
```

---

# 🔧 深入理解：Accelerate 底层原理

## Accelerator 的核心职责

```
Accelerator
    │
    ├── 检测环境（单卡/多卡/TPU/...）
    │
    ├── prepare() 方法
    │   ├── 包装 Model → DistributedDataParallel
    │   ├── 包装 DataLoader → 添加 DistributedSampler
    │   └── 包装 Optimizer → 处理梯度累积
    │
    ├── backward() 方法
    │   └── 自动梯度同步
    │
    └── 进程管理
        ├── is_main_process  # 是否是主进程
        ├── process_index    # 当前进程编号
        └── num_processes    # 总进程数
```

## accelerator.prepare() 内部做了什么？

### 1. 包装模型

```python
# prepare() 内部逻辑（简化版）
def prepare_model(model):
    # 将模型移到当前 GPU
    model = model.to(accelerator.device)
    
    # 用 DistributedDataParallel 包装
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[accelerator.local_process_index],
        output_device=accelerator.local_process_index
    )
    
    return model
```

### 2. 包装 DataLoader

```python
def prepare_dataloader(dataloader):
    # 替换采样器为 DistributedSampler
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=accelerator.num_processes,  # GPU 数量
        rank=accelerator.process_index           # 当前 GPU 编号
    )
    
    # 创建新的 DataLoader
    new_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,  # 🔥 关键：使用分布式采样器
        collate_fn=dataloader.collate_fn
    )
    
    return new_dataloader
```

### 3. 自动移动数据到 GPU

```python
# prepare() 后的 DataLoader 会自动加入数据移动逻辑
class AcceleratedDataLoader:
    def __iter__(self):
        for batch in self.original_dataloader:
            # 🔥 自动将每个 tensor 移到正确的 GPU
            yield move_to_device(batch, accelerator.device)
```

## accelerator.backward() 内部做了什么？

```python
def backward(self, loss):
    # 1. 如果使用梯度累积，需要缩放 loss
    if self.gradient_accumulation_steps > 1:
        loss = loss / self.gradient_accumulation_steps
    
    # 2. 计算梯度
    loss.backward()
    
    # 3. 🔥 多卡情况下，梯度会自动通过 DDP 同步
    # DDP 会在 backward 时自动触发 all-reduce 操作
    # 将所有 GPU 的梯度求平均
```

## 梯度同步原理（All-Reduce）

```
GPU 0: grad = [1.0, 2.0, 3.0]
GPU 1: grad = [2.0, 3.0, 4.0]
GPU 2: grad = [3.0, 4.0, 5.0]
GPU 3: grad = [4.0, 5.0, 6.0]

       ↓  All-Reduce (求和 + 平均)

所有 GPU: grad = [2.5, 3.5, 4.5]  # (1+2+3+4)/4, (2+3+4+5)/4, (3+4+5+6)/4

# 这样每个 GPU 上的模型参数更新是一致的！
```

---

## 📊 DataLoader 参数详解

```python
DataLoader(
    dataset,
    batch_size=32,          # 每个 batch 的样本数
    shuffle=True,           # 是否打乱（单卡用）
    num_workers=4,          # 数据加载的并行进程数
    pin_memory=True,        # 🔥 加速 GPU 数据传输
    drop_last=True,         # 丢弃最后不完整的 batch
    prefetch_factor=2,      # 每个 worker 预取的 batch 数
    persistent_workers=True # 保持 worker 进程存活
)
```

### 关键参数解释

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `num_workers` | CPU 并行加载数据 | CPU 核数 / GPU 数 |
| `pin_memory` | 数据放入锁页内存，加速 GPU 传输 | 总是 `True` |
| `drop_last` | 避免最后 batch 大小不一致 | 训练时 `True` |
| `prefetch_factor` | 预加载减少 GPU 等待 | 2-4 |

---

## 🧪 实战示例：完整的多卡训练代码

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, notebook_launcher

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 768)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'x': self.data[idx], 'y': self.labels[idx]}

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 10)
    
    def forward(self, x):
        return self.fc(x)

def train():
    # 1. 初始化 Accelerator
    accelerator = Accelerator()
    
    # 2. 准备数据和模型
    dataset = SimpleDataset(10000)
    loader = DataLoader(dataset, batch_size=32)  # 每卡 32
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 3. 🔥 Accelerate 包装
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    # 4. 训练循环
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            outputs = model(batch['x'])
            loss = criterion(outputs, batch['y'])
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    
    # 5. 保存模型
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), "model.pth")

# 启动
notebook_launcher(train, num_processes=4)
```

---

## 📖 参考资料

- [PyTorch DataLoader 官方文档](https://pytorch.org/docs/stable/data.html)
- [HuggingFace Accelerate 文档](https://huggingface.co/docs/accelerate)
- [PyTorch 分布式训练教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
