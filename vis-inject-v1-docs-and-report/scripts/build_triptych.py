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


def load_rgb(path, resize_to=None):
    img = Image.open(path).convert("RGB")
    if resize_to is not None and img.size != resize_to:
        # Resize (not crop) so the whole image is shown — matches the
        # 224x224 view the VLM actually saw at inference time.
        img = img.resize(resize_to, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    # The adversarial image is 224x224 because that's what the VLM was
    # fed; the clean original is the full-resolution screenshot. To make
    # the triptych an apples-to-apples comparison (same content the VLM
    # saw on both sides), resize the clean image down to the adv size
    # rather than corner-cropping it.
    adv_pil = Image.open(EXAMPLES / "adv_url_3m_ORIGIN_code.png").convert("RGB")
    adv = np.asarray(adv_pil, dtype=np.float32) / 255.0
    clean = load_rgb(EXAMPLES / "clean_ORIGIN_code.png", resize_to=adv_pil.size)

    assert clean.shape == adv.shape, (
        f"shape mismatch after resize: clean={clean.shape}, adv={adv.shape}"
    )

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
