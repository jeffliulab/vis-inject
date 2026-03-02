# demo_S2_AnyAttack

Reproduction of **"AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models"** (CVPR 2025).

- Paper: https://arxiv.org/abs/2410.05346
- Original code: https://github.com/jiamingzhang94/AnyAttack

## Overview

AnyAttack trains a **Decoder** network that generates adversarial perturbations for **any** input image, making it appear as a specified target image to Vision-Language Models. The attack transfers across multiple VLMs without requiring access to any specific target model during training.

## Architecture

```
Target Image ──> CLIP ViT-B/32 (frozen) ──> Embedding (512-dim)
                                                │
                                                v
                                           Decoder (~10M params)
                                                │
                                                v
                                        Noise (224×224×3)
                                                │
                                         clamp [-ε, ε]
                                                │
Clean Image ──────────────────────────> (+) ──> Adversarial Image
```

### Decoder Architecture

- **Input**: 512-dim CLIP embedding
- **FC**: 512 → 256 × 14 × 14
- **4× (ResBlock + UpBlock)**: 256 → 128 → 64 → 32 → 16
  - Each ResBlock includes EfficientAttention (linear-complexity spatial self-attention)
- **Head**: Conv2d(16 → 3), raw output (clamped externally)
- **Parameters**: ~10M

## Mathematical Principles

### Phase 1: Self-supervised Pre-training (LAION-Art)

The Decoder F learns to generate adversarial noise such that the CLIP embedding of the perturbed image matches the original image's embedding.

**InfoNCE contrastive loss with K-augmentation:**

Given a batch of images {x_i}, extract embeddings e_i = E(x_i), generate noise δ_i = F(e_i), then create K adversarial copies by adding noise to shuffled images:

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\text{sim}(e_i, \bar{e}'_i) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(e_i, \bar{e}'_j) / \tau)}$$

where $\bar{e}'_i = \frac{1}{K}\sum_{k=1}^{K} E(x_{\sigma_k(i)} + \delta_i)$ averages K shuffled adversarial embeddings, and τ is an annealing temperature (1.0 → 0.07).

### Phase 2: Fine-tuning (COCO)

Adapt the Decoder with additional auxiliary encoder losses for cross-model transferability:

$$\mathcal{L}_{total} = \mathcal{L}_{CLIP} + \mathcal{L}_{EVA} + \mathcal{L}_{ImageNet}$$

Each loss term uses either BiContrastiveLoss (bidirectional image-text contrastive) or DirectMatchingLoss (cosine similarity maximization).

## Training Pipeline

### Step 1: Download LAION-Art dataset

```bash
# Submit download job (see demos/LAION_ART_DATA/)
sbatch ../LAION_ART_DATA/download_laion_art.sh
```

### Step 2: Pre-train on LAION-Art (~3-5 hours on 1× H200)

```bash
python pretrain.py --tar-dir /path/to/LAION_ART/webdataset --epochs 5

# Or via SLURM:
sbatch hpc_train.sh pretrain
```

### Step 3: Fine-tune on COCO (~1-2 hours on 1× H200)

```bash
python finetune.py --pretrain-checkpoint checkpoints/pre-trained.pt --use-auxiliary

# Or via SLURM:
sbatch hpc_train.sh finetune
```

### Step 4: Generate adversarial images

```bash
# Single pair
python demo.py --decoder-path checkpoints/finetuned.pt \
               --clean-image dog.jpg --target-image cat.jpg

# Batch generation
python generate_adv.py --decoder-path checkpoints/finetuned.pt \
                       --clean-dir data/clean --target-dir data/target
```

### Step 5: Evaluate

```bash
python evaluate.py --adv-image adversarial.png \
                   --target-image cat.jpg --clean-image dog.jpg \
                   --target-vlms qwen2_5_vl_3b blip2_opt_2_7b
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 16/255 | L∞ perturbation budget |
| `clip_model` | ViT-B/32 | CLIP surrogate encoder |
| `batch_size` (pretrain) | 600 | Pre-training batch size |
| `batch_size` (finetune) | 100 | Fine-tuning batch size |
| `chunk` | 5 | K-augmentation copies |
| `lr` | 1e-4 | Learning rate (both phases) |
| `criterion` | BiContrastiveLoss | Fine-tuning loss function |

## File Structure

```
demo_S2_AnyAttack/
├── config.py            # All configuration parameters
├── requirements.txt     # Python dependencies
├── models/
│   ├── __init__.py
│   ├── clip_encoder.py  # CLIP ViT-B/32 wrapper (via open_clip)
│   └── decoder.py       # Decoder with ResBlock + EfficientAttention
├── losses.py            # DynamicInfoNCELoss, BiContrastiveLoss, DirectMatchingLoss
├── dataset.py           # LAION-Art WebDataset, COCO, ImageFolder loaders
├── pretrain.py          # Self-supervised pre-training on LAION-Art
├── finetune.py          # Fine-tuning on COCO with auxiliary encoders
├── generate_adv.py      # Batch adversarial image generation
├── evaluate.py          # Evaluation against target VLMs
├── demo.py              # Quick single-pair demo
└── hpc_train.sh         # SLURM job script (pretrain/finetune/generate/evaluate)
```

## Differences from Original Paper

| Aspect | Original | This Reproduction |
|--------|----------|-------------------|
| Pre-training data | LAION-400M (400M images) | LAION-Art (8M images) |
| Pre-training GPUs | 3-4× A100 | 1× H200 |
| Fine-tuning GPUs | 2× GPU (DDP) | 1× H200 |
| CLIP loading | Bundled OpenAI CLIP | open_clip library |
| Evaluation VLMs | BLIP/BLIP2/InstructBLIP/MiniGPT-4 | Qwen2.5-VL-3B, BLIP-2, + extensible |

## References

```bibtex
@inproceedings{zhang2025anyattack,
  title={Anyattack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models},
  author={Zhang, Jiaming and Ye, Junhong and Ma, Xingjun and Li, Yige and Yang, Yunfan and Chen, Yunhao and Sang, Jitao and Yeung, Dit-Yan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```
