# VisInject: 向图片注入 Prompt 以劫持多模态大模型

> **English version**: [README.md](README.md)

"病毒图片"是包含注入 prompt 的污染图片，人眼难以辨别，但能攻击图片并引导具有 MLLM 核心的 AI Agent（如 OpenClaw）做出有害行为。本项目研究如何制造这种"病毒图片"的对抗样本，并讨论如何防御。

项目目标：
1. 制造对抗样本并讨论防御策略。
2. 讨论病毒图片对 MLLM 和 AI Agent 的影响。
3. 评估开源和闭源 MLLM 的防御能力。
4. 讨论 AI 安全及闭源 MLLM 的防御方法。

## 1. 项目概述

### 1.1 目标

VisInject 将对抗性 prompt 嵌入图片，使扰动对人眼不可感知，但能让视觉语言模型（VLM）产生攻击者指定的输出。注入的 prompt 直接编码在自然图片的像素值中，推理时无需修改文本查询。

### 1.2 问题描述

现代 VLM（BLIP-2、Qwen2.5-VL、DeepSeek-VL、LLaVA 等）通过视觉编码器处理图片，然后将视觉 token 送入语言模型。这创造了一个攻击面：精心制作的像素扰动可以操控视觉 token 的表示，将语言模型的输出引导到攻击者选定的目标，实质上是在图片中注入了一条不可见的指令。

### 1.3 威胁模型

- **攻击者能力**: 攻击者可以在图片被目标 VLM 处理前修改像素。攻击者不控制文本查询。
- **攻击者目标**: 强制 VLM 产生特定目标输出（如伪造描述、注入指令响应或越狱绕过），不论用户问什么问题。
- **约束**: 扰动必须有界（L-inf 范数，通常 16/255 到 32/255），使图片在视觉上对人类保持不变。

### 1.4 研究演进

项目经历了三个阶段：

```
阶段 1: 单模型 PGD 攻击            阶段 2: 生成式方法                阶段 3: 最终整合
┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│ Demo_0  CLIP ViT     │     │ Demo_S1  StegoEncoder    │     │                  │
│ Demo1   BLIP-2       │ ──> │ Demo_S2  AnyAttack       │ ──> │    VisInject     │
│ Demo2   DeepSeek-VL  │     │ Demo_S3  UniversalAttack │     │ UniversalAttack  │
│ Demo3   Qwen2.5-VL   │     └──────────────────────────┘     │ + AnyAttack      │
└──────────────────────┘                                       └──────────────────┘
```

---

## 2. Demo 总结

> 每个 demo 的详细文档请见 [`demos/README_CN.md`](demos/README_CN.md)。

### 2.1 Demo_0: CLIP 跨模态嵌入攻击

- **模型**: CLIP ViT-L/14 | **方法**: PGD 对齐视觉和文本嵌入
- **结果**: CLIP 相似度 0.126 -> 0.68，PSNR 22-27 dB
- **结论**: CLIP 内有效，但**不迁移**到其他 VLM

### 2.2 Demo1: BLIP-2 端到端 PGD

- **架构**: Image -> EVA-ViT-G -> Q-Former -> OPT-2.7B -> CE Loss -> PGD
- **结果**: CE Loss 10.7 -> ~0.0，ASR 100%（含 QAA），PSNR ~32.5 dB
- **结论**: 首次成功的端到端 VLM 攻击

### 2.3 Demo2: DeepSeek-VL PGD

- **架构**: Image -> SigLIP-L -> MLP -> LLaMA-1.3B -> CE Loss -> PGD
- **结论**: PGD 攻击**与架构无关** — 跨编码器和 LLM 通用

### 2.4 Demo3: Qwen2.5-VL PGD

- **架构**: Image -> ViT-L -> PatchMerger -> Qwen2.5-3B -> CE Loss -> PGD
- **结果**: CE Loss 10.74 -> 0.00（~50步），所有 demo 中收敛最快

### 2.5 Demo_S1: StegoEncoder（已放弃）

- **方法**: 训练 U-Net + DCT 中频约束 | **结果**: ASR 0% | **结论**: 训练太慢

### 2.6 Demo_S2: AnyAttack (CVPR 2025)

- **AnyAttack_LAIONArt** (`demo_S2_AnyAttack/`): 自训练版本，数据下载中 (~12%)
- **AnyAttack_LAION400M** (`demo_S2P/`): 官方预训练权重，可直接使用
- **架构**: 目标图 -> CLIP ViT-B/32 -> 512维 -> Decoder (~28M) -> 噪声 (224x224)

### 2.7 Demo_S3: UniversalAttack

- **方法**: AdamW 直接像素优化，支持多模型联合攻击
- **论文 ASR**: 15-81%（视模型和攻击模式而定）

---

## 3. 对比分析

### 3.1 阶段 1: PGD 攻击

| 特性 | Demo_0 (CLIP) | Demo1 (BLIP-2) | Demo2 (DeepSeek) | Demo3 (Qwen) |
|------|:------------:|:--------------:|:----------------:|:------------:|
| 视觉编码器 | CLIP ViT-L/14 | EVA-ViT-G | SigLIP-L | ViT-L + Merger |
| 语言模型 | 无 | OPT-2.7B | LLaMA-1.3B | Qwen2.5-3B |
| 输入尺寸 | 224x224 | 224x224 | 384x384 | 392x392 |
| 显存 | ~1.6 GB | ~6 GB | ~5-8 GB | ~12 GB |
| 跨模型迁移 | 否 | 否 | 否 | 否 |

**结论**: PGD 攻击在单模型上有效，但本质上是模型特定的。

### 3.2 阶段 2: 生成式方法

| 特性 | Demo_S1 (StegoEncoder) | Demo_S2 (AnyAttack) | Demo_S3 (UniversalAttack) |
|------|:---------------------:|:-------------------:|:------------------------:|
| 方法 | 训练 U-Net 编码器 | 训练 CLIP 代理 Decoder | 优化单张图片 |
| 参数量 | ~55M | ~28M | 仅图片像素 |
| 跨模型 | 目标但未实现 | 通过 CLIP 代理 | 单模型或多模型 |
| 状态 | 已放弃 | 进行中 / 就绪 | 代码完成 |

---

## 4. 最终 VisInject 架构

### 4.1 核心思路

最终系统组合了 UniversalAttack 和 AnyAttack：

- **UniversalAttack 产生"做什么"**: 给定目标 prompt，像素优化生成一张抽象对抗图，将指令编码到视觉特征中迫使 VLM 服从。
- **AnyAttack 产生"怎么做"**: Decoder 取抽象图的 CLIP 嵌入，生成有界扰动，叠加到任意自然图片上，使其携带相同的对抗指令 — 同时对人眼完全自然。

### 4.2 流水线

```
目标 Prompt --> [UniversalAttack: 多模型联合像素优化] --> 通用对抗图 (448x448)
                                                                │
                                                     [CLIP ViT-B/32 编码]
                                                                │
                                                     [AnyAttack Decoder]
                                                                │
                                                           噪声 (224x224)
                                                                │
                                                干净图片 + 噪声 --> VisInject 图片
```

### 4.3 组合优势

| 挑战 | 仅 UniversalAttack | 仅 AnyAttack | 组合 |
|------|---------------------|--------------|------|
| Prompt 编码 | 抽象图编码任意 prompt | 无法编码任意 prompt | UniversalAttack 编码 |
| 视觉自然度 | 抽象图不自然 | 产生自然扰动 | AnyAttack 隐藏编码 |
| 迁移性 | 单/多模型 | 通过 CLIP 代理 | 双重机制跨模型 |
| 灵活性 | 任意 prompt，固定图片 | 任意图片对，固定目标 | 任意 prompt + 任意载体图 |

### 4.4 实现

最终流水线位于 [`visinject/`](visinject/)：

| 文件 | 说明 |
|------|------|
| `config.py` | 统一配置（目标 5 个 VLM 家族：Qwen、BLIP-2、DeepSeek、LLaVA、Phi） |
| `pipeline.py` | 端到端流水线：`target_phrase + clean_image -> 对抗图片` |
| `generate.py` | 独立 AnyAttack 融合（复用已有通用对抗图） |
| `evaluate.py` | 综合评估（ASR、CLIP 相似度、VLM 字幕、PSNR、Decoder 对比） |
| `web_demo.py` | Gradio Web 界面（中英文双语） |
| `hpc_pipeline.sh` | HPC SLURM 批处理脚本 |

**快速开始**:
```bash
# 完整流水线
python visinject/pipeline.py --target-phrase "Sure, here it is" \
                             --clean-images demos/demo_images/ORIGIN_dog.png \
                             --evaluate

# Web 界面
python visinject/web_demo.py --lang cn
```

---

## 5. 实现状态

### 5.1 已完成

| 组件 | 状态 | 关键结果 |
|------|------|----------|
| Demo_0: CLIP 嵌入攻击 | 完成 | 验证概念，发现迁移限制 |
| Demo1: BLIP-2 PGD | 完成 | 100% ASR，QAA 验证 |
| Demo2: DeepSeek-VL PGD | 完成 | 泛化到 SigLIP + LLaMA |
| Demo3: Qwen2.5-VL PGD | 完成 | 最快收敛，原生 forward() |
| Demo_S1: StegoEncoder | 完成（已放弃） | DCT 方法太慢，ASR 0% |
| Demo_S2: AnyAttack 代码 | 完成 | 训练/评估脚本就绪 |
| Demo_S2P: AnyAttack_LAION400M | 完成 | 官方权重已下载，推理就绪 |
| Demo_S3: UniversalAttack 代码 | 完成 | 6 个 VLM wrapper，多模型支持 |
| 模型注册表 | 完成 | 13+ VLM 集中管理 |
| HPC 基础设施 | 完成 | SLURM 脚本、日志、恢复支持 |
| **visinject/ 整合** | **完成** | **端到端流水线、Web 界面、评估** |

### 5.2 待完成

| 任务 | 依赖 |
|------|------|
| 在 HPC 上运行 UniversalAttack（多模型，3000步） | GPU 访问 |
| 用生成的通用对抗图运行 AnyAttack 融合 | UniversalAttack 输出 |
| 在 HPC 上运行完整评估（ASR、CLIP、字幕） | 对抗图片 |
| AnyAttack_LAIONArt: 完成 LAION-Art 下载（~12%） | 时间 |
| AnyAttack_LAIONArt: 预训练 + 微调 Decoder | 数据就绪 |
| Decoder 对比: LAION400M vs LAIONArt | 两个 Decoder 都就绪 |

---

## 6. 项目结构

```
VisInject/
├── visinject/              # 最终整合流水线
│   ├── config.py           # 统一配置
│   ├── pipeline.py         # 端到端：prompt + 图片 -> 对抗图片
│   ├── generate.py         # 独立 AnyAttack 融合
│   ├── evaluate.py         # 综合评估
│   ├── web_demo.py         # Gradio Web 界面（中英文）
│   └── hpc_pipeline.sh     # SLURM 批处理脚本
├── demos/                  # 实验 demo（阶段 1 & 2）
│   ├── demo_0_CLIP_ViT/
│   ├── demo1_BLIP2/
│   ├── demo2_DeepSeekVL_1/
│   ├── demo3_Qwen_2_5_VL_3B/
│   ├── demo_S1_Small_Model/
│   ├── demo_S2_AnyAttack/
│   ├── demo_S2P/
│   ├── demo_S3_UniversalAttack/
│   └── README.md / README_CN.md
├── model_registry.py       # VLM 集中注册表（13+ 模型）
├── README.md / README_CN.md
└── 自用说明书.txt
```

## 7. 硬件需求

| 组件 | 最低显存 | 推荐 GPU |
|------|----------|----------|
| AnyAttack 仅融合 | ~2 GB | 任意 GPU |
| UniversalAttack（单模型） | ~12 GB | RTX 3090+ |
| UniversalAttack（5模型联合） | ~37 GB | H200 / A100 80GB |
| 评估（每个 VLM） | ~4-14 GB | RTX 3090+ |
| Web 界面（快速模式） | ~2 GB | 任意 GPU |

所有实验在 Tufts HPC 集群的 NVIDIA H200 (80 GB HBM3) 上通过 SLURM 运行。

---

## 8. 参考文献

1. **AnyAttack** -- Zhang et al., "AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models," CVPR 2025. arXiv:2410.05346
2. **Universal Adversarial Attack** -- Schlarmann & Hein, "Universal Adversarial Attack on Aligned Multimodal LLMs," arXiv:2502.07987, 2025.
3. **BLIP-2** -- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," ICML 2023.
4. **Qwen2.5-VL** -- Bai et al., "Qwen2.5-VL Technical Report," 2025.
5. **DeepSeek-VL** -- Lu et al., "DeepSeek-VL: Towards Real-World Vision-Language Understanding," 2024.
6. **CLIP** -- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.
7. **PGD** -- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Examples," ICLR 2018.
