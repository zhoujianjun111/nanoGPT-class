# nanoGPT - 中文教学增强版

> 本版本为教育用途增强版，在原始 nanoGPT 基础上添加了中文古诗词与小说文本的训练案例，适合深度学习初学者学习实践。

---

##  重要更新 (Nov 2025)

nanoGPT 有一个更新、更强大的兄弟项目 [nanochat](https://github.com/karpathy/nanochat)。如果您想用于实际聊天应用，建议优先使用 nanochat。nanoGPT（本仓库）现已归档，但因其代码简洁、教学价值高，仍保留供学习参考。

---

##  项目简介

nanoGPT 是最简单、最快速的中等规模 GPT 训练/微调仓库。它是 [minGPT](https://github.com/karpathy/minGPT) 的重写版本，**优先考虑实用性而非理论教学**。

>  **形象比喻**：如果市面上的大模型是"航空母舰"，nanoGPT 就是一艘"游艇"——"麻雀虽小，五脏俱全"，非常适合初学者入门学习。

### 核心特点
- `train.py`：约 300 行的训练循环代码，清晰易懂
-  `model.py`：约 300 行的 GPT 模型定义，支持加载 OpenAI GPT-2 权重
-  单张 8×A100 40GB 节点，约 4 天即可复现 GPT-2 (124M) 在 OpenWebText 上的训练
- 代码极简，易于修改、从头训练或微调预训练模型

---

## 安装依赖

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

| 依赖包         | 用途                            |
| -------------- | ------------------------------- |
| `pytorch`      | 深度学习框架                    |
| `numpy`        | 数值计算                        |
| `transformers` | 加载 HuggingFace GPT-2 检查点   |
| `datasets`     | 下载与预处理 OpenWebText 数据集 |
| `tiktoken`     | OpenAI 的高效 BPE 分词器        |
| `wandb`        | （可选）训练日志记录            |
| `tqdm`         | 进度条显示                      |

---

## 教学案例：从零训练中文 GPT

本案例将训练两个模型：
1.  **诗词生成器**：使用 58,000 首唐诗训练歌词/诗词生成 GPT
2. 天龙八部风格生成器**：使用约 124 万字符的《天龙八部》文本训练武侠风格 GPT

###  案例目标
1. 掌握如何使用 PyTorch 构建 nanoGPT 模型网络，理解核心组件与原理
2. 学会数据准备、预处理、模型配置、训练流程及优化技巧
3. 掌握模型推理与文本采样的完整流程

---

###  案例准备

#### 1. 下载项目源码
```bash
# 推荐方式
git clone https://github.com/karpathy/nanoGPT.git

# 或下载 ZIP 后解压
```

#### 2. 准备训练数据
将以下数据文件放入 `nanoGPT/data/` 目录：
- `tang_poet.txt`：唐诗数据集，约 5.8 万首
- `tianlong.txt`：《天龙八部》文本，约 124 万字符

#### 3. 创建数据预处理脚本

以诗词数据为例，在 `nanoGPT/data/poemtext/` 目录下创建 `prepare.py`：

```python
import os
import tiktoken
import numpy as np

# 1. 读取原始文本
input_file_path = 'tang_poet.txt'
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# 2. 按 9:1 划分训练集/验证集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 3. 使用 GPT-2 BPE 编码器分词
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 4. 保存为二进制文件（uint16 节省空间）
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

运行预处理：
```bash
cd nanoGPT/data/poemtext
python prepare.py
```
成功后将生成 `train.bin` 和 `val.bin`，用于模型训练。

---

### 模型训练

#### 1. 创建训练配置文件

在 `nanoGPT/config/` 目录下创建 `train_poemtext_char.py`：

```python
# 输出目录 & 数据集路径
out_dir = 'out-poemtext-char'
dataset = 'poemtext'

# 评估与日志
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

# 数据参数
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # 上下文长度：256 个字符

# 模型架构（小型 Transformer）
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# 优化器参数
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
```

#### 2. 执行训练命令

🔹 **使用 GPU（推荐）**：
```bash
cd nanoGPT
python train.py config/train_poemtext_char.py
```

🔹 **使用 CPU（无显卡时）**：
```bash
python train.py config/train_poemtext_char.py \
    --device=cpu --compile=False --eval_iters=20 \
    --log_interval=1 --block_size=64 --batch_size=12 \
    --n_layer=4 --n_head=4 --n_embd=128 \
    --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

 **Apple Silicon Mac 用户**：添加 `--device=mps` 启用 Metal 加速 ⚡

#### 3. 训练输出
控制台将显示 `iter`, `loss`, `time` 等训练进度

训练完成后，最佳模型权重保存在 `out-poemtext-char/ckpt.pt`

---

### 🔍 模型推理与采样

使用 `sample.py` 从训练好的模型生成文本：

**GPU 推理**：

```bash
python sample.py --out_dir=out-poemtext-char
```

 **CPU 推理**：
```bash
python sample.py --out_dir=out-poemtext-char --device=cpu
```

 **指定起始文本采样**：
```bash
python sample.py --out_dir=out-poemtext-char \
    --start="床前明月光" \
    --num_samples=5 --max_new_tokens=100
```

**从文件读取提示词**：

```bash
python sample.py --out_dir=out-poemtext-char --start=FILE:prompt.txt
```

示例输出（诗词风格）：
```
春风拂柳绿，夜月照花红。
相思无尽处，独倚小楼中。
```

---

## 

---

##  致谢

本项目的实验均由 [Lambda Labs](https://lambdalabs.com) 提供 GPU 支持，感谢赞助！🎉

---

> 📝 **许可证**：本项目遵循 MIT 许可证，欢迎用于学习与研究。商用请遵守相关模型许可协议。
