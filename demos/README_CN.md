# VisInject 实验 Demos

本目录包含 VisInject 项目的所有实验 demo，从基础的单模型 PGD 攻击逐步演进到高级的生成式对抗方法。

> **English version**: [README.md](README.md)

## 总览

| Demo | 方法 | 核心结果 | 结论 |
|------|------|----------|------|
| [demo_0](#demo_0-clip-跨模态嵌入攻击) | CLIP 嵌入空间 PGD | 相似度 0.126→0.68, PSNR 22-27dB | CLIP 内有效，**不迁移** |
| [demo1](#demo1-blip-2-端到端-pgd) | BLIP-2 端到端 PGD | CE 10.7→0.0, ASR 100% | 单模型攻击成功 |
| [demo2](#demo2-deepseek-vl-pgd) | DeepSeek-VL 端到端 PGD | 同上模式，SigLIP+LLaMA | PGD 跨架构通用 |
| [demo3](#demo3-qwen25-vl-pgd) | Qwen2.5-VL 端到端 PGD | Loss 10.74→0.00（~50步） | 最快收敛，最干净实现 |
| [demo_S1](#demo_s1-stegoencoder) | StegoEncoder (DCT U-Net) | Loss 10.70→10.12, ASR 0% | **已放弃** — 太慢 |
| [demo_S2](#demo_s2-anyattack_laionart) | AnyAttack 自训练 (CVPR'25) | 代码完成，数据下载 ~12% | 等待 LAION-Art 数据 |
| [demo_S2P](#demo_s2p-anyattack_laion400m) | AnyAttack 官方权重 | coco_bi.pt 就绪 | 用于最终 VisInject |
| [demo_S3](#demo_s3-universalattack) | Universal 像素优化 | 论文 ASR: 15-81% | 用于最终 VisInject |

## 研究演进路线

```
阶段 1: 单模型 PGD                   阶段 2: 生成式                    阶段 3: 最终整合
┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│ Demo_0  CLIP ViT     │     │ Demo_S1  StegoEncoder    │     │                  │
│ Demo1   BLIP-2       │ ──> │ Demo_S2  AnyAttack       │ ──> │    VisInject     │
│ Demo2   DeepSeek-VL  │     │ Demo_S3  UniversalAttack │     │ UniversalAttack  │
│ Demo3   Qwen2.5-VL   │     └──────────────────────────┘     │ + AnyAttack      │
└──────────────────────┘                                       └──────────────────┘
```

---

## Demo_0: CLIP 跨模态嵌入攻击

**目录**: [`demo_0_CLIP_ViT/`](demo_0_CLIP_ViT/)

**目的**: 使用 PGD 将干净图片的 CLIP 视觉嵌入与目标文本的 CLIP 文本嵌入对齐。

**方法**: 在 L-inf 约束下最小化 `1 - cos(E_v(x + delta), E_t(target_text))`。完全在 CLIP ViT-L/14 共享嵌入空间内操作。

| 参数 | 值 |
|------|-----|
| 模型 | CLIP ViT-L/14 |
| 输入 | 224 x 224 |
| 扰动 | L-inf, eps = 16/255 |
| 显存 | ~1.6 GB |
| 步数 | 500-1000，每次攻击 ~22秒 |

**结果**:
- CLIP 相似度: **0.126 → 0.68**
- PSNR: 22-27 dB
- SSIM: 0.50-0.73

**结论**: 跨模态对齐在 CLIP 内有效，但**无法迁移**到其他 VLM（BLIP-2、DeepSeek、Qwen），因为它们使用不同的视觉编码器（EVA-CLIP、SigLIP 等）。这促使了后续 demo 中端到端攻击方法的探索。

---

## Demo1: BLIP-2 端到端 PGD

**目录**: [`demo1_BLIP2/`](demo1_BLIP2/)

**目的**: 首次成功的端到端 VLM 攻击。梯度从交叉熵损失流经整个 BLIP-2 流水线回传到输入像素。

**架构**: `图片 → EVA-ViT-G → Q-Former (32 queries) → 线性投影 → OPT-2.7B → CE Loss → PGD`

| 参数 | 值 |
|------|-----|
| 模型 | BLIP-2-OPT-2.7B (`Salesforce/blip2-opt-2.7b`) |
| 输入 | 224 x 224 |
| 扰动 | L-inf, eps = 32/255 |
| 显存 | ~6 GB |

**结果**:
- CE loss: **10.7 → ~0.0**
- ASR: **100%**（直接张量和 PNG 往返均成功，使用 QAA）
- PSNR: ~32.5 dB

**结论**: 端到端 PGD 攻击在单模型上非常有效。需要手动进行嵌入拼接以确保梯度通过 Q-Former 流动。

---

## Demo2: DeepSeek-VL PGD

**目录**: [`demo2_DeepSeekVL_1/`](demo2_DeepSeekVL_1/)

**目的**: 验证 PGD 攻击在不同 VLM 架构上的通用性（SigLIP 编码器 + LLaMA 骨干，对比 BLIP-2 的 EVA-CLIP + OPT）。

**架构**: `图片 → SigLIP-L (576 patches) → MLP 对齐器 (1024→2048) → LLaMA-1.3B → CE Loss → PGD`

| 参数 | 值 |
|------|-----|
| 模型 | DeepSeek-VL-1.3B (`deepseek-ai/deepseek-vl-1.3b-chat`) |
| 输入 | 384 x 384 |
| 扰动 | L-inf, eps = 32/255 |
| 显存 | ~5-8 GB |

**结果**: 攻击成功。手动构造嵌入拼接确保了完整的梯度流。

**结论**: PGD 攻击模式**与架构无关** — 在不同视觉编码器（EVA-CLIP vs SigLIP）和 LLM 骨干（OPT vs LLaMA）上均可工作。需要 `pip install deepseek-vl`。

---

## Demo3: Qwen2.5-VL PGD

**目录**: [`demo3_Qwen_2_5_VL_3B/`](demo3_Qwen_2_5_VL_3B/)

**目的**: 最干净、最快的 PGD 攻击实现。使用 Qwen2.5-VL 原生 `model.forward(labels=...)` 配合可微分的 `pixel_values`。

**架构**: `图片 → ViT-L (32层, 784 patches) → PatchMerger (2x2, 196 tokens) → Qwen2.5-3B → CE Loss → PGD`

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-VL-3B-Instruct (`Qwen/Qwen2.5-VL-3B-Instruct`) |
| 输入 | 392 x 392 |
| 扰动 | L-inf, eps = 32/255 |
| 显存 | ~12 GB |

**结果**:
- CE loss: **10.74 → 0.00**（~50步）
- ASR: **100%**（含 QAA）
- 攻击时间: 每张图片 ~2-3 分钟
- 所有 demo 中收敛最快

**结论**: 无需手动组装嵌入 — Qwen 原生前向传播处理一切。PGD 攻击的最佳开发体验。

---

## Demo_S1: StegoEncoder

**目录**: [`demo_S1_Small_Model/`](demo_S1_Small_Model/) | **状态**: 已放弃

**目的**: 训练轻量级 U-Net（~55M 参数）生成以目标 prompt 为条件的对抗扰动，使用 DCT 中频约束实现跨模型迁移。

**架构**: `干净图片 + Prompt → U-Net → 原始噪声 → DCT 滤波器 (频段 3-15) → L-inf 裁剪 → 对抗图片`

**核心创新**: DCT 中频域（8x8 块的频段 3-15）作为通用跨模型通道。所有视觉编码器都必须处理中频信息，而高频伪影因模型而异。

| 参数 | 值 |
|------|-----|
| 网络 | U-Net, 4 级 (64/128/256/512 通道), ~55M 参数 |
| 目标 VLM | BLIP-2, DeepSeek-VL, Qwen2.5-VL（同时攻击） |
| 训练 | 1500 epochs，单张图片 |

**结果**:
- CE loss: **10.70 → 10.12**（1500 epochs 后）
- ASR: **0%**
- 估计收敛: 每张图片 ~1400+ epochs

**结论**: **已放弃。** 从头训练生成模型的速度比直接 PGD 慢几个数量级。DCT 约束严重限制了攻击能力。被 Demo_S2（AnyAttack）和 Demo_S3（UniversalAttack）取代，后者使用已发表论文中验证过的架构。

---

## Demo_S2: AnyAttack_LAIONArt

**目录**: [`demo_S2_AnyAttack/`](demo_S2_AnyAttack/) | **论文**: [CVPR 2025](https://arxiv.org/abs/2410.05346) | **状态**: 等待数据

**目的**: 在 LAION-Art 数据集上自训练 AnyAttack Decoder 网络，与官方 LAION-400M 权重进行效果对比。

**架构**: `目标图片 → CLIP ViT-B/32 (冻结) → 512维 → Decoder (~28M 参数) → 噪声 (224x224)`

**两阶段训练**:
1. 在 LAION-Art（~800万图片）上自监督预训练，使用 InfoNCE 对比损失
2. 在 COCO 上微调，使用 BiContrastiveLoss + 辅助编码器（EVA02-Large, ViT-B/16）

| 参数 | 值 |
|------|-----|
| 代理模型 | CLIP ViT-B/32 (冻结) |
| Decoder | FC + 4x (ResBlock + EfficientAttention + Upsample), ~28M 参数 |
| 预训练数据 | LAION-Art (~83万张已下载, 约占 800万的 12%) |
| eps | 16/255 |

**当前状态**: 代码已完成。LAION-Art 数据集下载进行中（~12%）。数据量充足（~8万+）后将开始训练。结果将与 AnyAttack_LAION400M 对比。

---

## Demo_S2P: AnyAttack_LAION400M

**目录**: [`demo_S2P/`](demo_S2P/) | **状态**: 就绪

**目的**: 仅推理的 demo，使用 AnyAttack 的**官方预训练权重**（HuggingFace `jiamingzz/anyattack`）。在 LAION-400M 上预训练 + 在 COCO 上微调。无需训练。

**工作原理**:
1. 目标图片 → CLIP ViT-B/32 → 512维嵌入（空间信息丢失）
2. Decoder(嵌入) → 噪声图案 (224x224, 与目标无视觉相似性)
3. 对抗图片 = 干净图片 + clamp(噪声, -eps, eps)

Decoder 从**语义嵌入**生成噪声，而非视觉外观。人眼看到干净图片；CLIP 看到目标。

| 参数 | 值 |
|------|-----|
| 检查点 | `coco_bi.pt` (335 MB, 已下载) |
| 输入 | 干净图片 (224x224) + 目标图片 (224x224) |
| 输出 | 对抗图片 (224x224, 视觉上 ≈ 干净图片) |
| eps | 16/255 |

**结论**: 开箱即用。作为最终 VisInject 流水线的融合组件。

---

## Demo_S3: UniversalAttack

**目录**: [`demo_S3_UniversalAttack/`](demo_S3_UniversalAttack/) | **论文**: [arXiv 2502.07987](https://arxiv.org/abs/2502.07987) | **状态**: 就绪

**目的**: 优化一张**通用对抗图片**，迫使任意 MLLM 对任何问题都回答目标短语（如 "Sure, here it is"）。

**方法**: 通过 AdamW 直接优化像素。不训练神经网络。
- 参数化: `z = clip(z0 + gamma * tanh(z1), 0, 1)`，其中 z1 为可训练参数
- 损失: 目标 token 上的 masked 交叉熵，对所有目标模型求和
- 支持**多模型联合攻击**以实现跨架构迁移

| 参数 | 值 |
|------|-----|
| 可训练参数 | 仅图片像素 (z1 张量) |
| 优化器 | AdamW, lr = 0.01 |
| 步数 | 2000 (单模型) / 3000 (多模型) |
| gamma | 0.1 (单模型) / 0.5 (多模型) |
| 支持的 VLM | Qwen, BLIP-2, DeepSeek, LLaVA, Phi, Llama (6 个家族) |

**论文结果** (SafeBench ASR %):
| 模型 | 单模型 | 多答案 |
|------|--------|--------|
| Phi-3.5 | 15% | 81.3% |
| Llama-3.2-11B | 15% | 70.4% |
| Qwen2-VL-2B | 21.4% | 79.3% |
| LLaVA-1.5-7B | 44% | 46% |

**结论**: 有效的通用攻击方法。多模型联合优化实现跨架构迁移。作为最终 VisInject 流水线中 prompt→image 的组件。

---

## 辅助目录

| 目录 | 用途 |
|------|------|
| [`data_preparation/`](../data_preparation/) | 数据下载脚本（LAION-Art 数据集、VLM 模型、demo 图片）。从 `LAION_ART_DATA/` 迁移。详见 [`data_preparation/README.md`](../data_preparation/README.md)。 |
| `demo_images/` | 源图片（猫、狗、kpop、bill）和各 demo 生成的对抗样本。 |
| `demo_screenshots/` | 展示攻击结果的截图和演示材料。 |
| `demos_presentation/` | 自动生成的项目 PowerPoint 演示文稿。 |
| `demo_web.py` | Gradio Web UI，支持 BLIP-2、DeepSeek-VL、Qwen2.5-VL 推理和动态模型切换。 |

## 硬件需求

| Demo | 最低显存 | 推荐 GPU |
|------|----------|----------|
| Demo_0 | 2 GB | 任意 GPU |
| Demo1 | 6 GB | RTX 3060+ |
| Demo2 | 6 GB | RTX 3060+ |
| Demo3 | 12 GB | RTX 3090 / A100 |
| Demo_S1 | 14 GB | RTX 4090 / A100 |
| Demo_S2 (训练) | 20 GB | A100 / H200 |
| Demo_S2P (推理) | 2 GB | 任意 GPU |
| Demo_S3 (单模型) | 12 GB | RTX 3090+ |
| Demo_S3 (多模型, 5个) | 37 GB | H200 / A100 80GB |

所有 demo 均在 Tufts HPC 集群的 NVIDIA H200 (80 GB HBM3) 上通过 SLURM 测试。

## 共享基础设施

- **模型注册表** ([`model_registry.py`](../model_registry.py)): 集中管理 13+ 个 VLM，包含 HuggingFace ID、图片尺寸、归一化参数、显存估计。添加新模型只需添加一个字典条目。
- **通用 PGD 流程** (Demo_0 到 Demo3): 加载 VLM → 分词目标 → PGD 循环（前向传播, CE loss, 反向传播, 符号更新, 裁剪）→ QAA → 保存 PNG。

## 最终 VisInject 流水线

最终系统（位于 [`../visinject/`](../visinject/)）组合了 UniversalAttack + AnyAttack_LAION400M：

```
目标 Prompt ──> [UniversalAttack: 多模型联合像素优化] ──> 通用对抗图
                                                                │
                                                     [CLIP ViT-B/32 编码]
                                                                │
                                                     [AnyAttack Decoder]
                                                                │
                                                              噪声
                                                                │
                                                干净图片 + 噪声 ──> VisInject 图片
```
