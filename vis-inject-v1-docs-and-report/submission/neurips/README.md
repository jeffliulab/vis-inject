# `submission/neurips/` --- NeurIPS 2026 format build of the VisInject paper

This folder is a **reformatted-only** copy of the final report. The
scientific content is the same as [`report/xarchive-report/paper/`](../../xarchive-report/paper/);
only the LaTeX scaffolding has been rewritten to match the NeurIPS 2026
template (`neurips_2026.sty` + `checklist.tex`, both included verbatim
from the conference style package).

## What is here

| Path | What it is |
|---|---|
| `main.tex` | Document shell: `\documentclass{article}` + `\usepackage[preprint]{neurips_2026}` + the abstract + section includes + the acknowledgements environment + the bibliography + the checklist + the appendix. |
| `preamble_extras.tex` | Additional packages (graphicx, booktabs, listings, tikz, hyperref, amsmath, …). Layered on top of `neurips_2026.sty`; does not override its layout. |
| `neurips_2026.sty` | Verbatim from the NeurIPS 2026 conference style package. **Do not modify.** |
| `checklist.tex` | The mandatory NeurIPS Paper Checklist, filled in with paper-specific answers. |
| `refs.bib` | Verbatim copy of `xarchive-report/paper/refs.bib`. |
| `sections/0?_*.tex`, `A_*.tex`, `B_*.tex`, `C_*.tex` | Verbatim copies of the xarchive section files. The only thing the LaTeX format change touched is the title block, abstract environment, and acknowledgement environment in `main.tex`; section bodies are unchanged. |
| `figures/pipeline.tex` | A one-line forwarder that `\input`s the TikZ pipeline figure from `xarchive-report/paper/figures/pipeline.tex` (single source of truth — figures are not duplicated). |
| `Makefile` | `make` builds `main.pdf` via `latexmk -pdf -bibtex`. |
| `.gitignore` | Ignores LaTeX aux files; whitelists `main.pdf`. |

Image assets (PNGs, the per-VLM bar chart, the triptych) are **not**
copied — the `\graphicspath` in `preamble_extras.tex` points at
`../../xarchive-report/paper/figures/` so the build pulls them straight
from the xarchive folder. Editing or moving the xarchive figures will
affect this build.

## Build

```bash
cd report/submission/neurips
make
# → main.pdf in this directory
```

Build prerequisites: TeX Live or MacTeX with `pdflatex`, `latexmk`,
`bibtex`, and the `tikz`, `booktabs`, `subcaption`, `enumitem`,
`microtype`, `nicefrac` packages (all in standard distributions).

## Track / mode

The current `\usepackage[preprint]{neurips_2026}` line in `main.tex`
produces a **non-anonymous, no-line-numbers, arXiv-friendly** build with
the author block and artefact URLs visible. To switch:

| Goal | Edit in `main.tex` |
|---|---|
| Main Track, blind submission | Replace with `\usepackage{neurips_2026}` |
| Main Track, camera-ready | `\usepackage[main, final]{neurips_2026}` |
| Evaluations & Datasets Track, blind | `\usepackage[eandd]{neurips_2026}` |
| Evaluations & Datasets Track, camera-ready | `\usepackage[eandd, final]{neurips_2026}` |

Note: switching to a blind track (default, `[eandd]`) will:
- replace the author block with "Anonymous Author(s)",
- add line numbers,
- hide the `\begin{ack} … \end{ack}` block,
- but **will not** anonymise the URLs in the body
  (`huggingface.co/datasets/jeffliulab/visinject`, etc.) or in the
  acknowledgements --- you must do that by hand if the chosen track
  forbids identifying URLs.

## Differences vs `xarchive-report/paper/`

The change is structural, not content-level. Concretely:

- **Document class & style.** `xarchive` uses `\documentclass[11pt,letterpaper]{article}` plus a hand-rolled `preamble.tex` with custom navy-accent title, custom margins, and `titlesec`. This build uses `\documentclass{article}` plus `neurips_2026.sty`, which sets the page geometry (5.5 in × 9 in text), font (Times), section styling, abstract environment, and first-page footer.
- **Title block.** Replaced with a NeurIPS-compliant `\title{…}` + `\author{… \And …}` form; the colored navy-accent treatment of the xarchive build is dropped (the conference style has its own typography).
- **Abstract.** Wrapped in `\begin{abstract} … \end{abstract}` (NeurIPS expects this environment; the `neurips_2026.sty` typesets it centred and bolded above the body).
- **Acknowledgements.** Moved into the `\begin{ack} … \end{ack}` environment provided by the style file (auto-hidden under blind submission).
- **Checklist.** Added `checklist.tex` with paper-specific answers; required for any NeurIPS submission.
- **Section bodies.** Verbatim copies of the xarchive section files. No paragraph or sentence has been edited.

## Packaging

For NeurIPS submission, the upload is typically a ZIP containing
`main.tex`, `main.pdf`, `neurips_2026.sty`, `preamble_extras.tex`,
`checklist.tex`, `refs.bib`, `sections/`, and `figures/` (with the
referenced PNGs/PDFs from `../../xarchive-report/paper/figures/` copied
in). The build that lives in this folder uses cross-folder references
for convenience; for the actual upload the safest move is to rsync the
referenced figures into a packaged copy.
