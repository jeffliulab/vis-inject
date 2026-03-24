# VisInject

End-to-end adversarial prompt injection pipeline for multimodal LLMs.

> **Chinese version**: [README_CN.md](README_CN.md)

## Overview

VisInject generates adversarial images that look natural to humans but hijack VLM responses. It combines two techniques:

1. **UniversalAttack**: Optimizes a single universal adversarial image via multi-model joint pixel optimization. The image forces any target VLM to respond with a specified phrase regardless of the user's question.

2. **AnyAttack_LAION400M**: Encodes the universal image's semantics via CLIP and decodes noise through a pretrained Decoder. The noise is applied to clean images, making them visually unchanged but adversarially potent.

```
Target Prompt --> [UniversalAttack: multi-model pixel optimization] --> Universal Image
                                                                              |
                                                                   [CLIP ViT-B/32 encode]
                                                                              |
                                                                   [AnyAttack Decoder]
                                                                              |
                                                                           Noise
                                                                              |
                                                        Clean Image + Noise --> VisInject Image
```

## Quick Start

### Full pipeline (UniversalAttack + fusion + evaluation)

```bash
python pipeline.py --target-phrase "Sure, here it is" \
                   --clean-images ../demos/demo_images/ORIGIN_dog.png \
                   --evaluate
```

### Fusion only (reuse existing universal image)

```bash
python pipeline.py --universal-image outputs/universal/universal_final.png \
                   --clean-images ../demos/demo_images/ORIGIN_dog.png
```

### Standalone fusion

```bash
python generate.py --universal-image outputs/universal/universal_final.png \
                   --clean-images img1.png img2.png img3.png
```

### Evaluation

```bash
python evaluate.py --adv-images outputs/adversarial/adv_dog.png \
                   --clean-images ../demos/demo_images/ORIGIN_dog.png \
                   --universal-image outputs/universal/universal_final.png \
                   --compare-decoders
```

### Web demo

```bash
python web_demo.py --lang en          # English UI
python web_demo.py --lang cn          # Chinese UI
python web_demo.py --share            # Public link
```

### HPC (SLURM)

```bash
sbatch hpc_pipeline.sh full           # Full pipeline
sbatch hpc_pipeline.sh inject         # Fusion only
sbatch hpc_pipeline.sh eval           # Evaluation only
sbatch hpc_pipeline.sh compare        # Decoder comparison
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Unified configuration (UniversalAttack + AnyAttack + evaluation) |
| `utils.py` | Shared utilities (image loading, decoder loading, metrics) |
| `pipeline.py` | End-to-end pipeline: prompt + image -> adversarial image |
| `generate.py` | Standalone AnyAttack fusion (requires existing universal image) |
| `evaluate.py` | Comprehensive evaluation (ASR, CLIP, captions, PSNR, decoder comparison) |
| `web_demo.py` | Gradio web UI (bilingual EN/CN) |
| `hpc_pipeline.sh` | HPC SLURM batch script |

## Configuration

Edit `config.py` to customize:

- **ATTACK_TARGETS**: List of target VLMs for joint optimization. Enable/disable by commenting.
- **UNIVERSAL_ATTACK_CONFIG**: Optimization steps, gamma, quantization robustness, etc.
- **ANYATTACK_CONFIG**: Decoder path, CLIP model, eps budget.
- **EVAL_CONFIG**: Which VLMs to evaluate against, number of test questions.

### Adding a new VLM

1. Add a REGISTRY entry in `model_registry.py` (project root)
2. Implement an `MLLMWrapper` subclass in `demos/demo_S3_UniversalAttack/models/`
3. Add the model key to `ATTACK_TARGETS` in `config.py`

## Hardware Requirements

| Mode | VRAM | Recommended GPU |
|------|------|----------------|
| Fusion only (`generate.py`) | ~2 GB | Any GPU |
| Full pipeline (default 5 models) | ~37 GB | H200 / A100 80GB |
| Evaluation (per VLM) | ~4-14 GB | RTX 3090+ |
| Web demo (quick mode) | ~2 GB | Any GPU |

## Dependencies

- PyTorch, torchvision
- transformers, open_clip
- gradio (for web demo)
- deepseek-vl (for DeepSeek family only)
