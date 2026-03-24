# VisInject

面向多模态大模型的端到端对抗性 Prompt 注入流水线。

> **English version**: [README.md](README.md)

## 概述

VisInject 生成对人眼自然、但能劫持 VLM 回答的对抗图片。它组合了两种技术：

1. **UniversalAttack**: 通过多模型联合像素优化，生成一张通用对抗图片。该图片迫使任意目标 VLM 对任何问题回答指定短语。

2. **AnyAttack_LAION400M**: 通过 CLIP 编码通用对抗图的语义，再由预训练 Decoder 解码生成噪声。将噪声叠加到干净图片上，视觉几乎不变但具有对抗效果。

```
目标 Prompt --> [UniversalAttack: 多模型联合像素优化] --> 通用对抗图
                                                                │
                                                     [CLIP ViT-B/32 编码]
                                                                │
                                                     [AnyAttack Decoder]
                                                                │
                                                              噪声
                                                                │
                                                干净图片 + 噪声 --> VisInject 图片
```

## 快速开始

### 完整流水线（UniversalAttack + 融合 + 评估）

```bash
python pipeline.py --target-phrase "Sure, here it is" \
                   --clean-images ../demos/demo_images/ORIGIN_dog.png \
                   --evaluate
```

### 仅融合（复用已有通用对抗图）

```bash
python pipeline.py --universal-image outputs/universal/universal_final.png \
                   --clean-images ../demos/demo_images/ORIGIN_dog.png
```

### 独立融合

```bash
python generate.py --universal-image outputs/universal/universal_final.png \
                   --clean-images img1.png img2.png img3.png
```

### 评估

```bash
python evaluate.py --adv-images outputs/adversarial/adv_dog.png \
                   --clean-images ../demos/demo_images/ORIGIN_dog.png \
                   --universal-image outputs/universal/universal_final.png \
                   --compare-decoders
```

### Web 界面

```bash
python web_demo.py --lang cn          # 中文界面
python web_demo.py --lang en          # 英文界面
python web_demo.py --share            # 公开链接
```

### HPC (SLURM)

```bash
sbatch hpc_pipeline.sh full           # 完整流水线
sbatch hpc_pipeline.sh inject         # 仅融合
sbatch hpc_pipeline.sh eval           # 仅评估
sbatch hpc_pipeline.sh compare        # Decoder 对比
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 统一配置（UniversalAttack + AnyAttack + 评估参数） |
| `utils.py` | 工具函数（图片读写、Decoder 加载、指标计算） |
| `pipeline.py` | 端到端流水线：prompt + 图片 -> 对抗图片 |
| `generate.py` | 独立 AnyAttack 融合（需要已有通用对抗图） |
| `evaluate.py` | 综合评估（ASR、CLIP 相似度、VLM 字幕、PSNR、Decoder 对比） |
| `web_demo.py` | Gradio Web 界面（中英文双语） |
| `hpc_pipeline.sh` | HPC SLURM 批处理脚本 |

## 配置

编辑 `config.py` 自定义：

- **ATTACK_TARGETS**: 联合优化的目标 VLM 列表。通过注释启用/禁用。
- **UNIVERSAL_ATTACK_CONFIG**: 优化步数、gamma、量化鲁棒性等。
- **ANYATTACK_CONFIG**: Decoder 路径、CLIP 模型、eps 预算。
- **EVAL_CONFIG**: 评估用 VLM、测试问题数量。

### 添加新 VLM

1. 在 `model_registry.py`（项目根目录）中添加 REGISTRY 条目
2. 在 `demos/demo_S3_UniversalAttack/models/` 中实现 `MLLMWrapper` 子类
3. 在 `config.py` 的 `ATTACK_TARGETS` 中添加模型 key

## 硬件需求

| 模式 | 显存 | 推荐 GPU |
|------|------|----------|
| 仅融合 (`generate.py`) | ~2 GB | 任意 GPU |
| 完整流水线（默认 5 模型） | ~37 GB | H200 / A100 80GB |
| 评估（每个 VLM） | ~4-14 GB | RTX 3090+ |
| Web 界面（快速模式） | ~2 GB | 任意 GPU |

## 依赖

- PyTorch, torchvision
- transformers, open_clip
- gradio（Web 界面）
- deepseek-vl（仅 DeepSeek 家族需要）
