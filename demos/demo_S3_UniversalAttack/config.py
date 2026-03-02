"""
demo_S3_UniversalAttack Configuration
======================================
Reproduces "Universal Adversarial Attack on Aligned Multimodal LLMs"
(arXiv 2502.07987, 2025).

Paper: https://arxiv.org/abs/2502.07987
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_registry import init_model_env

init_model_env()

# ── Attack parameters ─────────────────────────────────────────────
ATTACK_CONFIG = {
    # Target models (from model_registry keys)
    # Single-model: ["qwen2_5_vl_3b"]
    # Multi-model:  ["qwen2_5_vl_3b", "phi_3_5_vision", "llava_1_5_7b"]
    "target_models": ["qwen2_5_vl_3b"],

    # Perturbation constraint: gamma * tanh(z1)
    # Auto-adjusted: 0.1 for single-model, 0.5 for multi-model (if not set)
    "gamma": None,  # None = auto
    "gamma_single": 0.1,
    "gamma_multi": 0.5,

    # Image parameters
    "image_size": (448, 448),  # Typical MLLM input resolution

    # Optimization
    "lr": 1e-2,
    "num_steps": 2000,
    "optimizer": "AdamW",

    # Target response
    "target_phrase": "Sure, here it is",

    # Quantization robustness
    "quant_robustness": True,

    # Gaussian blur on perturbation
    "gaussian_blur": False,
    "blur_kernel_size": 5,
    "blur_sigma": 1.0,

    # Multi-answer attack
    "multi_answer": False,
    "answer_pool": [
        "Sure, here it is",
        "Of course, I can help with that",
        "Absolutely, here you go",
        "Yes, I'll provide that information",
        "Sure thing, let me explain",
    ],

    # Localization attack
    "localize": False,
    "localize_scale_min": 0.5,
    "localize_scale_max": 0.9,
}

# ── Output / logging ─────────────────────────────────────────────
OUTPUT_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "output_dir": "outputs",
    "log_dir": "logs",
    "save_every": 200,      # Save image every N steps
    "log_every": 50,         # Print loss every N steps
}
