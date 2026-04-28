"""Build report/pdf/figures/per_vlm.pdf — per-VLM Output-Affected score bar chart.

Run from the repo root::

    python report/scripts/build_per_vlm_bar.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJ_ROOT / "report" / "pdf" / "figures" / "per_vlm.pdf"

# Source: docs/experiment_report.md §6.3 (verified).
DATA = [
    ("Qwen2.5-VL-3B",    8.45, "#1A3A5C"),
    ("Qwen2-VL-2B",      8.34, "#2E86DE"),
    ("DeepSeek-VL-1.3B", 8.19, "#2E86DE"),
    ("BLIP-2-OPT-2.7B",  0.00, "#C0392B"),
]


def main():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    labels = [d[0] for d in DATA]
    values = [d[1] for d in DATA]
    colors = [d[2] for d in DATA]

    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_xlabel("Output-Affected score (max 10)", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", va="center", ha="left", fontsize=10,
            color="#1F2937",
        )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(PROJ_ROOT)}")


if __name__ == "__main__":
    main()
