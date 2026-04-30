# `xarchive-report/` — arXiv Preprint Planning Workspace

> Standalone planning folder for the arXiv preprint version of VisInject.
> Independent of the course-final-report subtree (`report/`, gitignored).
> Created: 2026-04-30.

## Why this folder exists

The course-final deliverables (PDF report + slide deck + HF dataset) are done and submitted. The professor suggested polishing this work into an **arXiv preprint** — leveraging the HF dataset's organic adoption (~300 downloads / first month) as evidence of demand.

This folder collects the research, audits, and roadmap for that arXiv release. **Nothing here ships to the public** — these are working documents.

## File index

| File | Content | Read first if you want to … |
|---|---|---|
| [`PLAN.md`](PLAN.md) | **Master roadmap** — recommended path, content keep/cut/add list, week-by-week timeline, risk assessment | …decide what to do next. **Start here.** |
| [`AUDIT.md`](AUDIT.md) | Honest audit of the v1.1 project: what's solid, what's thin, what reviewers will attack | …know which claims are defensible |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Concrete experiment specs (P0 / P1 / P2) with effort estimates | …see what new experiments are needed |
| [`RELATED_WORK.md`](RELATED_WORK.md) | 42-paper related-work survey, fully verified, organised by subarea | …write the paper's §2 |
| [`DATASET_LANDSCAPE.md`](DATASET_LANDSCAPE.md) | 30-HF-dataset survey + best-practice patterns + recommendations for `jeffliulab/visinject` | …upgrade the HF dataset card |
| [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) | Proposed arXiv paper section structure (8-9 pages) | …start writing the paper |

## TL;DR

- **Recommended path**: arXiv v1 in 3-4 weeks (Path A, see PLAN.md), with a stretch goal of an AISec / AdvML workshop submission in 2-3 months (Path B). Top conference push (Path C, 6-12 months) is **not recommended right now** — work is too thin without significant new experiments.
- **Minimum experiments needed**: 4 (BLIP-2 ablation, inter-rater study, judge keyword sensitivity, +1 frontier-model transfer). Total ~30 GPU-hours + ~25 human-hours. Doable in 2 weeks.
- **HF dataset must-fix**: viewer is broken (`StreamingRowsError`); judge results not uploaded; no flat parquet manifest. These are blocking issues for credibility.
- **Paper positioning**: lead with the dual-axis evaluation methodology (drift ≠ injection) and BLIP-2 immunity finding. Treat 0.227% injection rate as a **feature** (honest negative result), not a bug.

## Folder lifecycle

This folder will be **iteratively edited** as the arXiv release progresses. Once the preprint is submitted, this folder can either:
- (a) be kept as a working diary of the research path, OR
- (b) be archived to a v1.0-arxiv git tag and removed from main

Decision deferred until after submission.
