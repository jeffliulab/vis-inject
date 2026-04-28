"""Build report/pdf/figures/triptych.pdf — clean / adversarial / amplified-diff.

Stage 2 produces a perceptually invisible noise layer.  This figure shows:
  (a) the clean code screenshot used in Case Study A;
  (b) the adversarial version actually fed to the VLM;
  (c) the per-pixel difference, amplified 10x for readability.

Run from the repo root::

    python report/scripts/build_triptych.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = PROJ_ROOT / "outputs" / "succeed_injection_examples"
OUT = PROJ_ROOT / "report" / "pdf" / "figures" / "triptych.pdf"


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main():
    clean = load_rgb(EXAMPLES / "clean_ORIGIN_code.png")
    adv = load_rgb(EXAMPLES / "adv_url_3m_ORIGIN_code.png")

    if clean.shape != adv.shape:
        h = min(clean.shape[0], adv.shape[0])
        w = min(clean.shape[1], adv.shape[1])
        clean = clean[:h, :w]
        adv = adv[:h, :w]

    diff = adv - clean
    amp = 10.0
    diff_vis = np.clip(0.5 + amp * diff, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))
    titles = [
        "(a) Clean image",
        "(b) Adversarial image",
        f"(c) Difference, $\\times{int(amp)}$",
    ]
    for ax, img, title in zip(axes, [clean, adv, diff_vis], titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color="#1F2937", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(0.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(PROJ_ROOT)}")


if __name__ == "__main__":
    main()
