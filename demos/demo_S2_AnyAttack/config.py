"""
demo_S2_AnyAttack Configuration
================================
Reproduces "AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks
on Vision-language Models" (CVPR 2025).

Paper: https://arxiv.org/abs/2410.05346
Code:  https://github.com/jiamingzhang94/AnyAttack
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_registry import init_model_env

init_model_env()

# ── Attack parameters ─────────────────────────────────────────────
ATTACK_CONFIG = {
    "eps": 16 / 255,                 # L-inf perturbation budget
    "clip_model": "ViT-B/32",       # CLIP surrogate model
    "embed_dim": 512,                # CLIP ViT-B/32 embedding dimension
    "image_size": 224,               # Input image resolution
}

# ── Pre-training on LAION-Art ─────────────────────────────────────
PRETRAIN_CONFIG = {
    "mode": "self_pretrain",         # "self_pretrain" or "pretrained_weights"

    # self_pretrain settings
    "tar_dir": "/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset",
    "batch_size": 600,
    "lr": 1e-4,
    "epochs": 5,
    "chunk": 5,                      # K-augmentation: number of shuffled copies
    "checkpoint_every": 500,         # Save checkpoint every N batches
    "checkpoint_dir": "checkpoints",

    # pretrained_weights settings
    "pretrained_weights_url": "https://huggingface.co/Jiaming94/anyattack",
    "pretrained_weights_file": "checkpoints/pre-trained.pt",
}

# ── Fine-tuning on COCO ──────────────────────────────────────────
FINETUNE_CONFIG = {
    "dataset": "coco_retrieval",
    "data_dir": "data/mscoco",
    "imagenet_dir": "data/imagenet/train",
    "batch_size": 100,
    "lr": 1e-4,
    "epochs": 20,
    "criterion": "BiContrastiveLoss",  # "BiContrastiveLoss" or "Cosine"
    "use_auxiliary_encoders": True,     # EVA02-Large + ViT-B/16 for transferability
    "checkpoint_dir": "checkpoints",
    "pretrain_checkpoint": "checkpoints/pre-trained.pt",
}

# ── Adversarial image generation ─────────────────────────────────
GENERATE_CONFIG = {
    "decoder_path": "checkpoints/finetuned.pt",
    "output_dir": "outputs/adversarial",
    "batch_size": 250,
}

# ── Evaluation ────────────────────────────────────────────────────
EVAL_CONFIG = {
    "target_vlms": [
        "qwen2_5_vl_3b",
        "blip2_opt_2_7b",
    ],
    "eval_tasks": ["captioning", "retrieval"],
    "num_eval_samples": 500,
}

# ── Output / logging ─────────────────────────────────────────────
OUTPUT_CONFIG = {
    "log_dir": "logs",
    "results_dir": "outputs/results",
}
