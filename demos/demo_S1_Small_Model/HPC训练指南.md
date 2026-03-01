# HPC 训练指南 — demo_S1_Small_Model

## 目录
1. [训练目的](#训练目的)
2. [文件上传](#文件上传)
3. [环境配置](#环境配置)
4. [模型下载](#模型下载)
5. [数据准备](#数据准备)
6. [启动训练](#启动训练)
7. [监控训练进度](#监控训练进度)
8. [查看结果](#查看结果)
9. [常见问题](#常见问题)

---

## 训练目的

**训练一个叫 StegoEncoder 的小模型（~55M 参数），使其能把任意图像转换为带有隐藏攻击指令的版本。**

```
任意图像 → [StegoEncoder] → 外观几乎不变的图像 → 喂给 Qwen2.5-VL-3B → 输出 "VISINJECT_TRIGGERED"
```

这一阶段（Stage 1A）只针对 Qwen2.5-VL-3B 训练。完成后你将得到：
- 一个可以对**任意新图像**一次前向传播（毫秒级）生成攻击图的模型权重文件
- 在训练集图像上的预期 ASR（触发成功率）：**30-70%**
- 在未见图像上的预期 ASR：**10-40%**（泛化能力）

> **训练规模说明：** Stage 1A 需要约 1500 epoch × 50 张图。
> 每 epoch 对每张图做 1 次梯度更新（通过完整 Qwen 3B 模型的反向传播）。
> A100 40GB 上预计约 **30-40 小时**完成，对应本地 4090 约需 140 小时（不现实）。

---

## 文件上传

### 只需上传代码（不上传模型权重和数据集）

```
上传以下内容：
VisInject/
├── model_registry.py          ← 必须（统一模型管理）
└── demos/
    └── demo_S1_Small_Model/   ← 完整上传此文件夹
        ├── config.py
        ├── run_demo.py
        ├── prepare_data.py
        ├── requirements.txt
        ├── augmentation.py
        ├── losses.py
        ├── utils.py
        ├── evaluate.py
        ├── rewards.py
        ├── models/
        ├── training/
        ├── encoders/
        ├── prompts/
        └── vlms/

不需要上传（HPC 上重新生成）：
- model_cache/          ← HPC 上重新下载模型
- data/                 ← HPC 上重新下载 COCO
- logs_and_outputs/     ← 训练自动生成
- __pycache__/
- test/                 ← 可选，测试脚本
```

### 推荐用 git 同步代码

```bash
# 本地推送到远程 repo（如 GitHub）
git add .
git commit -m "stage1a ready for HPC"
git push

# HPC 上拉取
git clone https://github.com/yourname/VisInject.git
# 或
git pull
```

---

## 环境配置

### 推荐目录结构（在 HPC 上）

```
$HOME/                        ← 你的 home 目录（通常空间小，只放代码）
├── VisInject/                ← 代码仓库
└── envs/
    └── visinject/            ← conda 环境（用 --prefix 指定路径）

/path/to/large_storage/       ← 大存储（项目分配空间，几百 GB）
├── model_cache/              ← HuggingFace 模型缓存
└── visinject_data/           ← COCO 数据集
```

> **为什么要把 conda env 和 model_cache 放到大存储？**
> HPC 的 home 目录通常只有 20-50 GB，conda env (~5 GB) + 模型缓存 (~15 GB) 很容易撑满导致作业失败。

---

### Step 1：创建 conda 环境（指定路径）

```bash
# 查看你的大存储路径（各 HPC 不同，通常是 /scratch/yourid 或 /project/xxx）
echo $SCRATCH   # 或
df -h           # 看哪个分区空间大

# 在大存储创建 conda 环境（--prefix 指定路径，避免占满 home）
conda create --prefix /scratch/yourid/envs/visinject python=3.11 -y

# 激活（注意 --prefix 方式激活需要写完整路径）
conda activate /scratch/yourid/envs/visinject
```

### Step 2：安装 PyTorch（与 HPC CUDA 版本对应）

```bash
# 查看 HPC 的 CUDA 版本
nvcc --version    # 或
nvidia-smi        # 看右上角 CUDA Version

# CUDA 12.x（A100/H100 常见）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3：安装项目依赖

```bash
cd ~/VisInject/demos/demo_S1_Small_Model
pip install -r requirements.txt

# Qwen2.5-VL 专用工具（requirements.txt 可能未包含）
pip install qwen_vl_utils
```

### Step 4：设置 HuggingFace 缓存路径

**关键步骤，否则模型下载到 home 目录导致爆盘。**

```bash
# 创建模型缓存目录
mkdir -p /scratch/yourid/model_cache

# 设置环境变量（写入 ~/.bashrc 永久生效）
echo 'export HF_HOME=/scratch/yourid/model_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/scratch/yourid/model_cache/hub' >> ~/.bashrc
source ~/.bashrc

# 验证
echo $HF_HOME   # 应输出 /scratch/yourid/model_cache
```

> **说明：** 项目代码中的 `model_registry.py` 会调用 `init_model_env()`，
> 该函数使用 `os.environ.setdefault`，即**若系统已设置 `HF_HOME`，则保持不变**。
> 所以 `.bashrc` 里设置的路径会被优先使用。

---

## 模型下载

### 情况 A：HPC 计算节点有外网（直接下载）

训练时模型会自动下载，跳过此步骤。

### 情况 B：HPC 计算节点无外网（常见！需在登录节点预下载）

```bash
# 在登录节点（有网络）执行预下载
conda activate /scratch/yourid/envs/visinject
export HF_HOME=/scratch/yourid/model_cache

# 下载 Qwen2.5-VL-3B（Stage 1A 必需，~7GB）
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print('Downloading Qwen2.5-VL-3B-Instruct...')
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype='auto'
)
print('Done.')
"
```

### 情况 C：HPC 完全隔离（需从本地打包上传）

```bash
# 本地：打包 model_cache（约 15GB，较大）
cd /path/to/local/VisInject
tar -czf model_cache.tar.gz model_cache/

# 上传到 HPC
scp model_cache.tar.gz yourid@hpc.university.edu:/scratch/yourid/

# HPC 上解压
tar -xzf /scratch/yourid/model_cache.tar.gz -C /scratch/yourid/
```

---

## 数据准备

```bash
cd ~/VisInject/demos/demo_S1_Small_Model

# 下载 COCO val2017（~780 MB）并自动划分 train/test
python prepare_data.py

# 完成后目录结构：
# data/
# ├── train/   ← 4000 张 jpg（训练用）
# └── test/    ← 500 张 jpg（评估用）
```

> 若 HPC 计算节点无外网，在登录节点运行 `prepare_data.py` 即可，数据保存到本地磁盘，计算节点可直接读取。

---

## 启动训练

### 方式 A：SLURM 提交作业（推荐）

创建 `train_s1a.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=visinject_s1a
#SBATCH --partition=gpu             # 根据你的 HPC 修改分区名
#SBATCH --gres=gpu:a100:1           # 申请 1 块 A100（最低要求，40GB 足够）
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00             # 48 小时，留余量
#SBATCH --output=logs/slurm_%j.log  # 标准输出
#SBATCH --error=logs/slurm_%j.err   # 错误输出

# 激活环境
conda activate /scratch/yourid/envs/visinject

# 关键：设置 HuggingFace 缓存路径
export HF_HOME=/scratch/yourid/model_cache
export HF_HUB_CACHE=/scratch/yourid/model_cache/hub

# 关键：CUDA 内存优化（避免显存碎片导致速度下降 100 倍）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 进入项目目录
cd ~/VisInject/demos/demo_S1_Small_Model

# 启动 Stage 1A 训练
python run_demo.py --stage1a
```

提交作业：
```bash
mkdir -p logs
sbatch train_s1a.slurm

# 查看作业状态
squeue -u yourid

# 实时查看日志
tail -f logs/slurm_<jobid>.log
```

### 方式 B：交互式运行（短期调试用）

```bash
# 申请交互式 GPU 节点（时间较短）
srun --gres=gpu:a100:1 --mem=32G --time=2:00:00 --pty bash

conda activate /scratch/yourid/envs/visinject
export HF_HOME=/scratch/yourid/model_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/VisInject/demos/demo_S1_Small_Model
python run_demo.py --stage1a
```

### 调整训练规模

训练参数全在 `config.py` 的 `STAGE1A_CONFIG` 中：

```python
STAGE1A_CONFIG = {
    "lr": 1e-3,           # 学习率（不建议改动）
    "epochs": 1500,       # 训练总 epoch（默认 HPC 目标）
    "num_images": 50,     # 使用多少张训练图（默认 50）
    "save_interval": 50,  # 每 50 epoch 保存一次 checkpoint
    ...
}
```

也可以用命令行参数临时覆盖：
```bash
# 快速验证（10 epoch × 5 张图，约 10 分钟）
python run_demo.py --stage1a --num-images 5

# 正式训练（用 config.py 默认值）
python run_demo.py --stage1a
```

---

## 监控训练进度

### 查看训练日志

```bash
# 实时跟踪训练日志（最重要的文件）
tail -f logs_and_outputs/stage1a_e2e_*/train.log

# 示例正常输出：
# 2026-03-01 10:00:00 [INFO] Epoch   1/1500 | CE=10.6000 | Total=10.6001 | PSNR=29.7dB
# 2026-03-01 10:05:00 [INFO] Epoch  10/1500 | CE=10.2000 | Total=10.2001 | PSNR=30.2dB
# 2026-03-01 10:50:00 [INFO] Epoch  50/1500 | CE=9.8000 | Total=9.8001 | PSNR=29.5dB
```

### 判断训练是否正常

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| **CE Loss** | 持续从 ~10.6 下降 | 长时间无变化（卡在 10.6±0.1） |
| **PSNR** | 28-36 dB | < 20 dB（扰动过大）或 > 40 dB（扰动过小） |
| **GPU 利用率** | 90-99% | 长时间 0%（进程崩溃） |

```bash
# 实时查看 GPU 状态
watch -n 5 nvidia-smi

# 快速查看当前 CE Loss 趋势
grep "Epoch" logs_and_outputs/stage1a_e2e_*/train.log | tail -20
```

### 预期 CE Loss 下降曲线

```
Epoch   1:  CE ≈ 10.6  （起点）
Epoch  100:  CE ≈ 10.0
Epoch  300:  CE ≈ 9.0
Epoch  600:  CE ≈ 7.0
Epoch 1000:  CE ≈ 4.0
Epoch 1500:  CE ≈ 2.0  （开始能触发的边界）
```

> CE < 2.0 时，模型开始有概率触发 Qwen 输出目标词。
> CE < 0.5 时，触发率显著提升（类似 PGD 的 0.003 水平）。

---

## 查看结果

### 训练产物目录结构

训练完成后，`logs_and_outputs/` 下会生成：

```
logs_and_outputs/
└── stage1a_e2e_20260301_100000/      ← 时间戳命名
    ├── train.log                      ← 完整训练日志
    ├── history.json                   ← 每 epoch 的 loss 记录
    └── checkpoints/
        ├── best.pt                    ← 最优 checkpoint（最重要！）
        ├── epoch_0050.pt             ← 每 50 epoch 保存一个
        ├── epoch_0100.pt
        └── ...
```

**下载回本地只需要 `best.pt`（约 210 MB）。**

### Step 1：运行 ASR 评估（Test 10）

```bash
cd ~/VisInject/demos/demo_S1_Small_Model

# 评估训练集 ASR（验证模型是否学会攻击）
python test/10_test_vlm_trigger.py \
    --data-dir data/train \
    --n-test 20

# 评估测试集 ASR（验证泛化能力）
python test/10_test_vlm_trigger.py \
    --data-dir data/test \
    --n-test 20
```

**解读输出：**
```
ASR 矩阵
  [none           ]  8/20 =  40.0%  (目标 ≥ 10%)  ✓    ← 无失真触发率 40%
  [jpeg_q50       ]  5/20 =  25.0%  (目标 ≥ 5%)   ✓    ← JPEG 压缩后仍有 25%
  [screenshot_sim ]  4/20 =  20.0%  (目标 ≥ 3%)   ✓    ← 截图后仍有 20%

Stage 1A → Stage 1B 推进门槛检查
  [1] 'none' 条件至少触发 1 张: ✓
  [2] 至少 2 种失真条件有触发: ✓

✅ Stage 1A 通过！可推进 Stage 1B 三模型泛化测试
```

### Step 2：查看图像质量指标（Test 9）

```bash
python test/9_test_proxy_metrics.py --n-test 50
```

**关注指标：**
- `PSNR ≥ 28 dB`：图像质量合格（肉眼不易察觉扰动）
- `SSIM ≥ 0.85`：结构相似度合格
- `feature_stability = 1.0`：Qwen 特征在失真后保持稳定

### Step 3：单张图像推理演示

```bash
# 把你自己的图片转换为对抗图像
python run_demo.py --infer \
    --checkpoint logs_and_outputs/stage1a_e2e_xxx/checkpoints/best.pt \
    --image /path/to/your_photo.jpg \
    --output /path/to/output_adv.jpg

# 然后用 Qwen 测试触发效果（参考 test/10_test_vlm_trigger.py）
```

### Step 4：从 history.json 绘制 Loss 曲线（可选）

```python
import json, matplotlib.pyplot as plt

with open("logs_and_outputs/stage1a_e2e_xxx/history.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
ce_losses = [h["ce_loss"] for h in history]

plt.plot(epochs, ce_losses)
plt.xlabel("Epoch")
plt.ylabel("CE Loss")
plt.title("Stage 1A Training — CE Loss")
plt.axhline(y=2.0, color='r', linestyle='--', label='触发阈值 (~CE=2)')
plt.legend()
plt.savefig("loss_curve.png")
```

---

## 常见问题

### Q: 作业因 OOM（显存不足）中断了

**原因：** A100 40GB 通常够用，但某些图像分辨率异常大时可能超限。

**解决：**
```bash
# 1. 确认 PYTORCH_CUDA_ALLOC_CONF 已设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 申请更大显存的节点
#SBATCH --gres=gpu:a100_80g:1   # 80GB A100
```

### Q: 训练 loss 完全不动（一直是 10.6）

**原因：** 最常见的问题，通常是 `config.py` 里 epsilon 太小（应为 32/255）。

**检查：**
```bash
python -c "import config; print(config.STEGO_MODEL_CONFIG['epsilon'])"
# 应输出 0.12549... (32/255)，若是 0.0627... (16/255) 则需要修改
```

### Q: HPC 计算节点下载模型失败

**解决：** 在登录节点（有网络）预下载（见[模型下载](#模型下载)章节），或联系 HPC 管理员配置代理。

### Q: 如何中断训练并从 checkpoint 继续？

当前 `E2ETrainer` 不支持断点续训（每次从头开始）。

**临时方案：** 找到最近的 `epoch_xxxx.pt`，修改 `e2e_trainer.py` 加载该 checkpoint 继续训练（需要代码修改，可提 issue 或切到 Agent 模式让 AI 实现）。

### Q: 想用更多图像训练（>50 张）

修改 `config.py`：
```python
STAGE1A_CONFIG = {
    "num_images": 200,   # 改为 200 或更多
    ...
}
```
或使用命令行覆盖：
```bash
python run_demo.py --stage1a --num-images 200
```
更多图像 = 更好的泛化，但训练时间成比例增加。

---

## 快速参考

```bash
# 环境激活
conda activate /scratch/yourid/envs/visinject
export HF_HOME=/scratch/yourid/model_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/VisInject/demos/demo_S1_Small_Model

# 数据准备（只需一次）
python prepare_data.py

# 启动训练
python run_demo.py --stage1a

# 实时监控 loss
tail -f logs_and_outputs/stage1a_e2e_*/train.log | grep "Epoch"

# 查看结果
python test/10_test_vlm_trigger.py --n-test 20
python test/9_test_proxy_metrics.py --n-test 50
```
