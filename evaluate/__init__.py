"""
VisInject Stage 3: Evaluation module.

Two-step pipeline:
  Stage 3a (HPC, GPU)  — pairs.py: query target VLMs on (clean, adv) image pairs
                          and save responses as JSON.
  Stage 3b (local, API) — judge.py: GPT-4o / Claude scores each pair for
                          adversarial injection success.

Public API (re-exported here for backward compatibility with existing
imports in pipeline.py and demo/web_demo.py):
"""

from .pairs import (
    generate_response_pairs,
    run_evaluation,
    evaluate_asr,
    evaluate_image_quality,
    evaluate_clip,
    evaluate_captions,
)

__all__ = [
    "generate_response_pairs",
    "run_evaluation",
    "evaluate_asr",
    "evaluate_image_quality",
    "evaluate_clip",
    "evaluate_captions",
]
