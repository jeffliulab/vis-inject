# VisInject Demos

This directory contains all experimental demos for the VisInject project, progressing from basic single-model PGD attacks to advanced generative adversarial approaches.

> **Chinese version**: [README_CN.md](README_CN.md)

## Quick Summary

| Demo | Method | Key Result | Conclusion |
|------|--------|------------|------------|
| [demo_0](#demo_0-clip-cross-modal-embedding-attack) | CLIP embedding PGD | Sim 0.126→0.68, PSNR 22-27dB | Works within CLIP, **no transfer** |
| [demo1](#demo1-blip-2-end-to-end-pgd) | BLIP-2 E2E PGD | CE 10.7→0.0, ASR 100% | Single-model success |
| [demo2](#demo2-deepseek-vl-pgd) | DeepSeek-VL E2E PGD | Same pattern, SigLIP+LLaMA | PGD generalizes across architectures |
| [demo3](#demo3-qwen25-vl-pgd) | Qwen2.5-VL E2E PGD | Loss 10.74→0.00 in ~50 steps | Fastest convergence, cleanest impl |
| [demo_S1](#demo_s1-stegoencoder) | StegoEncoder (DCT U-Net) | Loss 10.70→10.12, ASR 0% | **Abandoned** — too slow |
| [demo_S2](#demo_s2-anyattack_laionart) | AnyAttack self-train (CVPR'25) | Code complete, data ~12% | Awaiting LAION-Art data |
| [demo_S2P](#demo_s2p-anyattack_laion400m) | AnyAttack official weights | coco_bi.pt ready | Used in final VisInject |
| [demo_S3](#demo_s3-universalattack) | Universal pixel optimization | Paper ASR: 15-81% | Used in final VisInject |

## Research Progression

```
Stage 1: Single-Model PGD            Stage 2: Generative           Stage 3: Final
┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│ Demo_0  CLIP ViT     │     │ Demo_S1  StegoEncoder    │     │                  │
│ Demo1   BLIP-2       │ ──> │ Demo_S2  AnyAttack       │ ──> │    VisInject     │
│ Demo2   DeepSeek-VL  │     │ Demo_S3  UniversalAttack │     │ UniversalAttack  │
│ Demo3   Qwen2.5-VL   │     └──────────────────────────┘     │ + AnyAttack      │
└──────────────────────┘                                       └──────────────────┘
```

---

## Demo_0: CLIP Cross-Modal Embedding Attack

**Directory**: [`demo_0_CLIP_ViT/`](demo_0_CLIP_ViT/)

**Purpose**: Align a clean image's CLIP visual embedding with a target text's CLIP text embedding using PGD.

**Method**: Minimize `1 - cos(E_v(x + delta), E_t(target_text))` under L-inf constraint. Operates entirely in CLIP ViT-L/14 shared embedding space.

| Parameter | Value |
|-----------|-------|
| Model | CLIP ViT-L/14 |
| Input | 224 x 224 |
| Perturbation | L-inf, eps = 16/255 |
| VRAM | ~1.6 GB |
| Steps | 500-1000, ~22s per attack |

**Results**:
- CLIP similarity: **0.126 → 0.68**
- PSNR: 22-27 dB
- SSIM: 0.50-0.73

**Conclusion**: Cross-modal alignment works within CLIP, but **does not transfer** to other VLMs (BLIP-2, DeepSeek, Qwen) because they use different vision encoders (EVA-CLIP, SigLIP, etc.). This motivated the end-to-end approach in subsequent demos.

---

## Demo1: BLIP-2 End-to-End PGD

**Directory**: [`demo1_BLIP2/`](demo1_BLIP2/)

**Purpose**: First successful end-to-end VLM attack. Gradients flow from cross-entropy loss through the entire BLIP-2 pipeline back to input pixels.

**Architecture**: `Image → EVA-ViT-G → Q-Former (32 queries) → Linear Proj → OPT-2.7B → CE Loss → PGD`

| Parameter | Value |
|-----------|-------|
| Model | BLIP-2-OPT-2.7B (`Salesforce/blip2-opt-2.7b`) |
| Input | 224 x 224 |
| Perturbation | L-inf, eps = 32/255 |
| VRAM | ~6 GB |

**Results**:
- CE loss: **10.7 → ~0.0**
- ASR: **100%** on both direct tensor and PNG round-trip (with QAA)
- PSNR: ~32.5 dB

**Conclusion**: End-to-end PGD attack is highly effective on a single model. Manual embedding packing required for gradient flow through Q-Former.

---

## Demo2: DeepSeek-VL PGD

**Directory**: [`demo2_DeepSeekVL_1/`](demo2_DeepSeekVL_1/)

**Purpose**: Verify that PGD attack generalizes to a different VLM architecture (SigLIP encoder + LLaMA backbone, vs EVA-CLIP + OPT in BLIP-2).

**Architecture**: `Image → SigLIP-L (576 patches) → MLP Aligner (1024→2048) → LLaMA-1.3B → CE Loss → PGD`

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-VL-1.3B (`deepseek-ai/deepseek-vl-1.3b-chat`) |
| Input | 384 x 384 |
| Perturbation | L-inf, eps = 32/255 |
| VRAM | ~5-8 GB |

**Results**: Successful attack. Hand-crafted embedding concatenation ensures full gradient flow through the model.

**Conclusion**: PGD attack pattern is **architecture-agnostic** — works across different vision encoders (EVA-CLIP vs SigLIP) and LLM backbones (OPT vs LLaMA). Requires `pip install deepseek-vl`.

---

## Demo3: Qwen2.5-VL PGD

**Directory**: [`demo3_Qwen_2_5_VL_3B/`](demo3_Qwen_2_5_VL_3B/)

**Purpose**: Cleanest and fastest PGD attack implementation. Uses Qwen2.5-VL's native `model.forward(labels=...)` with differentiable `pixel_values`.

**Architecture**: `Image → ViT-L (32 layers, 784 patches) → PatchMerger (2x2, 196 tokens) → Qwen2.5-3B → CE Loss → PGD`

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-VL-3B-Instruct (`Qwen/Qwen2.5-VL-3B-Instruct`) |
| Input | 392 x 392 |
| Perturbation | L-inf, eps = 32/255 |
| VRAM | ~12 GB |

**Results**:
- CE loss: **10.74 → 0.00** in ~50 steps
- ASR: **100%** with QAA
- Attack time: ~2-3 min per image
- Fastest convergence among all demos

**Conclusion**: No manual embedding assembly needed — Qwen's native forward pass handles everything. Best developer experience for PGD attacks.

---

## Demo_S1: StegoEncoder

**Directory**: [`demo_S1_Small_Model/`](demo_S1_Small_Model/) | **Status**: Abandoned

**Purpose**: Train a lightweight U-Net (~55M params) to produce adversarial perturbations conditioned on a target prompt, with DCT mid-frequency constraints for cross-model transferability.

**Architecture**: `Clean Image + Prompt → U-Net → Raw Noise → DCT Filter (bands 3-15) → L-inf Clamp → Adversarial Image`

**Key Innovation**: DCT mid-frequency domain (bands 3-15 of 8x8 blocks) as a universal cross-model channel. All vision encoders must process mid-frequencies, unlike high-frequency artifacts which are model-specific.

| Parameter | Value |
|-----------|-------|
| Network | U-Net, 4 scales (64/128/256/512 ch), ~55M params |
| Target VLMs | BLIP-2, DeepSeek-VL, Qwen2.5-VL (simultaneously) |
| Training | 1500 epochs, single image |

**Results**:
- CE loss: **10.70 → 10.12** after 1500 epochs
- ASR: **0%**
- Estimated convergence: ~1400+ epochs per single image

**Conclusion**: **Abandoned.** Training a generative model from scratch is orders of magnitude slower than direct PGD. The DCT constraint severely limits attack capacity. Replaced by Demo_S2 (AnyAttack) and Demo_S3 (UniversalAttack), which use proven architectures from published papers.

---

## Demo_S2: AnyAttack_LAIONArt

**Directory**: [`demo_S2_AnyAttack/`](demo_S2_AnyAttack/) | **Paper**: [CVPR 2025](https://arxiv.org/abs/2410.05346) | **Status**: Awaiting Data

**Purpose**: Self-train the AnyAttack Decoder network on LAION-Art dataset for comparison with the official LAION-400M weights.

**Architecture**: `Target Image → CLIP ViT-B/32 (frozen) → 512-dim → Decoder (~28M params) → Noise (224x224)`

**Two-phase training**:
1. Self-supervised pre-training on LAION-Art (~8M images) with InfoNCE contrastive loss
2. Fine-tuning on COCO with BiContrastiveLoss + auxiliary encoders (EVA02-Large, ViT-B/16)

| Parameter | Value |
|-----------|-------|
| Surrogate | CLIP ViT-B/32 (frozen) |
| Decoder | FC + 4x (ResBlock + EfficientAttention + Upsample), ~28M params |
| Pre-training data | LAION-Art (~830K images downloaded, ~12% of 8M) |
| eps | 16/255 |

**Current Status**: Code complete. LAION-Art dataset download in progress (~12%). Training will begin when sufficient data is available (~80K+ images). Results will be compared against AnyAttack_LAION400M.

---

## Demo_S2P: AnyAttack_LAION400M

**Directory**: [`demo_S2P/`](demo_S2P/) | **Status**: Ready

**Purpose**: Inference-only demo using AnyAttack's **official pre-trained weights** from HuggingFace (`jiamingzz/anyattack`). Pre-trained on LAION-400M + fine-tuned on COCO. No training required.

**How it works**:
1. Target image → CLIP ViT-B/32 → 512-dim embedding (spatial info lost)
2. Decoder(embedding) → noise pattern (224x224, no visual resemblance to target)
3. Adversarial image = clean image + clamp(noise, -eps, eps)

The decoder generates noise from a **semantic embedding**, not from the visual appearance. Human eye sees the clean image; CLIP sees the target.

| Parameter | Value |
|-----------|-------|
| Checkpoint | `coco_bi.pt` (335 MB, downloaded) |
| Input | Clean image (224x224) + Target image (224x224) |
| Output | Adversarial image (224x224, visually ≈ clean) |
| eps | 16/255 |

**Conclusion**: Works out of the box. Used as the fusion component in the final VisInject pipeline.

---

## Demo_S3: UniversalAttack

**Directory**: [`demo_S3_UniversalAttack/`](demo_S3_UniversalAttack/) | **Paper**: [arXiv 2502.07987](https://arxiv.org/abs/2502.07987) | **Status**: Ready

**Purpose**: Optimize a **single universal adversarial image** that forces any MLLM to respond with a target phrase (e.g., "Sure, here it is") regardless of the user's question.

**Method**: Direct pixel optimization via AdamW. No neural network trained.
- Parameterization: `z = clip(z0 + gamma * tanh(z1), 0, 1)` where z1 is trainable
- Loss: Masked cross-entropy on target tokens, summed across all target models
- Supports **multi-model joint attack** for cross-architecture transferability

| Parameter | Value |
|-----------|-------|
| Trainable | Image pixels only (z1 tensor) |
| Optimizer | AdamW, lr = 0.01 |
| Steps | 2000 (single-model) / 3000 (multi-model) |
| gamma | 0.1 (single) / 0.5 (multi) |
| Supported VLMs | Qwen, BLIP-2, DeepSeek, LLaVA, Phi, Llama (6 families) |

**Paper results** (SafeBench ASR %):
| Model | Single | Multi-Answer |
|-------|--------|-------------|
| Phi-3.5 | 15% | 81.3% |
| Llama-3.2-11B | 15% | 70.4% |
| Qwen2-VL-2B | 21.4% | 79.3% |
| LLaVA-1.5-7B | 44% | 46% |

**Conclusion**: Effective universal attack. Multi-model joint optimization enables cross-architecture transferability. Used as the prompt→image component in the final VisInject pipeline.

---

## Supporting Directories

| Directory | Purpose |
|-----------|---------|
| [`data_preparation/`](../data_preparation/) | Download scripts for LAION-Art dataset, VLM models, and demo images. Moved from `LAION_ART_DATA/`. See [`data_preparation/README.md`](../data_preparation/README.md). |
| `demo_images/` | Source images (cat, dog, kpop, bill) and generated adversarial examples used across demos. |
| `demo_screenshots/` | Screenshots and presentation materials demonstrating attack results. |
| `demos_presentation/` | Auto-generated PowerPoint presentation of the project. |
| `demo_web.py` | Gradio web UI for inference on BLIP-2, DeepSeek-VL, Qwen2.5-VL. Supports dynamic model switching. |

## Hardware Requirements

| Demo | Min VRAM | Recommended GPU |
|------|----------|----------------|
| Demo_0 | 2 GB | Any GPU |
| Demo1 | 6 GB | RTX 3060+ |
| Demo2 | 6 GB | RTX 3060+ |
| Demo3 | 12 GB | RTX 3090 / A100 |
| Demo_S1 | 14 GB | RTX 4090 / A100 |
| Demo_S2 (train) | 20 GB | A100 / H200 |
| Demo_S2P (infer) | 2 GB | Any GPU |
| Demo_S3 (single) | 12 GB | RTX 3090+ |
| Demo_S3 (multi, 5 models) | 37 GB | H200 / A100 80GB |

All demos tested on NVIDIA H200 (80 GB HBM3) via SLURM on Tufts HPC cluster.

## Shared Infrastructure

- **Model Registry** ([`model_registry.py`](../model_registry.py)): Centralized registry for 13+ VLMs with HuggingFace IDs, image sizes, normalization params, VRAM estimates. Add a new model by adding one dict entry.
- **Common PGD Flow** (Demo_0 through Demo3): Load VLM → tokenize target → PGD loop (forward, CE loss, backprop, sign update, clamp) → QAA → save PNG.

## Final VisInject Pipeline

The final system (in [`../visinject/`](../visinject/)) combines UniversalAttack + AnyAttack_LAION400M:

```
Target Prompt ──> [UniversalAttack: multi-model pixel optimization] ──> Universal Image
                                                                              │
                                                                   [CLIP ViT-B/32 encode]
                                                                              │
                                                                   [AnyAttack Decoder]
                                                                              │
                                                                           Noise
                                                                              │
                                                        Clean Image + Noise ──> VisInject Image
```
