# VisInject — Final Report

This subtree contains the **course-final-project deliverables** for VisInject v1: a slide deck and a written report. It is intentionally separated from the project code (`src/`, `attack/`, `models/`, `evaluate/`), the project documentation (`docs/`), and project automation (`scripts/`).

## Layout

```
report/
├── scripts/                 Report-only build scripts
│   ├── build_slides.py      python-pptx → report/slides/VisInject_final.pptx
│   ├── build_triptych.py    matplotlib  → report/pdf/figures/triptych.pdf
│   └── build_per_vlm_bar.py matplotlib  → report/pdf/figures/per_vlm.pdf
├── slides/                  PPT output (.pptx is gitignored)
└── pdf/                     LaTeX project
    ├── main.tex             Entrypoint; \input{}s sections/*.tex
    ├── preamble.tex         Packages, macros, title block
    ├── refs.bib             Bibliography
    ├── sections/            01_intro.tex … 08_conclusion.tex
    ├── figures/             pipeline.tex (TikZ source) + generated PDFs
    ├── Makefile             make / make clean / make watch / make figures
    └── main.pdf             Compiled PDF (committed)
```

## Build the slides

```bash
python report/scripts/build_slides.py
# → report/slides/VisInject_final.pptx
```

Requires `python-pptx`. The script reads images from `data/images/`, `outputs/succeed_injection_examples/`, and `docs/HF-downloads.png` (paths relative to the repo root).

## Build the PDF

```bash
cd report/pdf
make figures   # regenerate matplotlib figures (only needed when data changes)
make           # latexmk -pdf -bibtex main.tex
```

Requires TeX Live or MacTeX (`pdflatex` + `latexmk` + `bibtex`) and Python 3 with `matplotlib` + `Pillow` for figure scripts.

Aux artifacts (`*.aux`, `*.log`, `*.bbl`, …) are ignored by the local `report/pdf/.gitignore`. Only `main.pdf` is tracked.

## Source-of-truth conventions

- Numerical results in slides and PDF must match [`docs/experiment_report.md`](../docs/experiment_report.md) (the canonical Chinese narrative). The plan file `~/.claude/plans/v2-v1-mian-ppt-pdf-plan-dreamy-twilight.md` lists the verified numbers.
- Case-study images come from [`outputs/succeed_injection_examples/manifest.json`](../outputs/succeed_injection_examples/manifest.json).
- The top-level `README.md` links here under **Final Report & Slides**.
