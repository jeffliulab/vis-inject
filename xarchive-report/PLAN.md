# VisInject arXiv Preprint — Master Plan

> **Audience**: you (Pang Liu), planning your first arXiv submission.
> **Status**: research complete (3 parallel agents reviewed 42 papers + 30 HF datasets + audited the local project). Decision pending.
> **Last updated**: 2026-04-30.

---

## 1. Context

Where we are:
- VisInject v1.1 (course final) is done — 18-page LaTeX report, 25-slide deck, public HF dataset, public HF Space, ~300 downloads in first month.
- Professor saw the work and suggested an arXiv preprint, framing the HF download trace as external interest signal.
- You have ~3-12 weeks of post-graduation runway before deciding whether to push toward a publication.

What this plan answers:
1. **What to keep** from the existing report
2. **What to expand** (sections, claims, related work)
3. **Whether new experiments are needed** — and which ones, with effort estimates
4. **Concrete week-by-week timeline**

The supporting documents in this folder (`AUDIT.md`, `EXPERIMENTS.md`, `RELATED_WORK.md`, `DATASET_LANDSCAPE.md`, `PAPER_OUTLINE.md`) carry the underlying detail.

---

## 2. Three Paths (pick one)

| | **Path A — arXiv-only release** | **Path B — arXiv + workshop submission** | **Path C — top-conf push** |
|---|---|---|---|
| **Effort** | 3-4 weeks | 2-3 months | 6-12 months |
| **New experiments** | 0-2 (HF dataset polish only) | 4-6 (must-do experiments from `EXPERIMENTS.md`) | All P0+P1+P2 (~5 attacks × 3 defenses + scaling + mechanistic analysis) |
| **Risk of "boring"** | Medium — work as-is reads thin | Low — workshop reviewers welcome focused empirical papers | High — need surprise + scale to clear top-conf bar |
| **Acceptance probability** | n/a (arXiv has no review) | 40-60% at AISec / AdvML workshop | <10% at NeurIPS / ICML main; 15-25% at NeurIPS D&B Track |
| **Realistic outcome** | "First arXiv preprint, citable, course project polished" | "AISec / AdvML / SaTML workshop paper, real publication credit" | "Top-tier conference paper if everything works; nothing if it falls through" |

### Recommended: **Path A as foundation + Path B as stretch goal**

Reasoning:
- Path A is **bounded**: 3-4 weeks gets you a polished, citable arXiv preprint. Even if you stop there, you have a permanent academic record of this work.
- Path B is a **natural extension** of Path A — same content, plus 4-6 specific experiments that cover the audit's most damning gaps. The investment is real (60-80 hours), but the payoff is a real publication.
- Path C is **not recommended now**. It requires V2-dev's 5-attack / 3-defense matrix (un-executed, ~3 months of HPC work alone), plus mechanistic analysis (gradient flow / feature space) that you currently don't have the pipeline for. Reconsider Path C only if Path B succeeds and you decide to extend.

The rest of this plan assumes you're committing to **Path A** with **Path B as a 60-day stretch goal**.

---

## 3. Content Decision: Keep / Strengthen / Add / Drop

### 3.1 Keep as-is (already paper-quality)

| Component | Why it's solid | Where it lives now |
|---|---|---|
| Three-stage pipeline architecture (UAA + AnyAttack fusion + dual-dim eval) | Composition is original; named building blocks are now properly attributed | PDF §3 |
| Dual-dimension evaluation (Output-Affected via LCS drift + Target-Injected via keyword regex) | The core methodological contribution. Cleanly separates drift from payload | PDF §3.4 |
| 6,615-pair systematic sweep across 7 prompts × 7 images × 4 VLMs × 3 ensembles | Largest publicly-released benign-prompt × adversarial-image evaluation matrix in the literature | PDF §4-§5, HF dataset |
| 7 attack prompts with motivation per category | Covers literal-text / semantic-class / brand / misinformation / ad / phishing / override | PDF §4.2 |
| 60-question pool with USER / AGENT / SCREENSHOT split, mapped back to threat scenarios | Good coverage justification | PDF Appendix A |
| Architecture finding: BLIP-2 fully immune (0/2,205) | Genuinely surprising; supports Q-Former-as-defense future work | PDF §5, §7 |
| Honest negative results: low injection rate, no GPT-4o transfer | Reframes field optimism — leads naturally to "drift ≠ injection" headline | PDF §5, §7 |
| Threat model with 3 operational scenarios | Concrete and well-motivated | PDF §2 |
| Two case-study response panels with red-highlighted injected content | Effective qualitative evidence | PDF §6 |

### 3.2 Strengthen (good but reviewer-vulnerable)

| Component | What's weak | How to fix |
|---|---|---|
| **Causal explanation for BLIP-2 immunity** | Three competing hypotheses (Stage-2 fusion / resolution downsampling / gradient dilution) but no ablation | Run direct $x_u \to$ BLIP-2 ablation (skip Stage 2). 4-6 GPU-hours. See `EXPERIMENTS.md` E1 |
| **Judge reliability** | No inter-rater study; keyword lists subjective | Re-grade 200-300 stratified subsample with second coder (human or different LLM); compute Cohen's $\kappa$. ~10 hours total. `EXPERIMENTS.md` E2 |
| **Keyword-list audit trail** | Lists hand-curated, listed in code comments only | Move to JSON config file with provenance per variant; ablate strict / baseline / loose lists. `EXPERIMENTS.md` E3 |
| **GPT-4o transfer claim** | N=1 manual sample; reads as anecdote | Systematic test: top-3 highest-ASR cases × 2 frontier models (GPT-4o + Gemini 2.0 Flash). ~$50 in API. `EXPERIMENTS.md` E4 |
| **"Architecture > size"** claim | Comparing 3B Qwen vs 2.7B BLIP-2 is not a clean size comparison; architecture confounded with size | Reframe to "Q-Former bottleneck provides robustness" — drop the size-vs-arch wording |
| **Image-set narrowness** | Only 7 images; all natural / screenshot. Reviewers will say "image-type-dependent" | Either expand to 21-30 images (`EXPERIMENTS.md` E5) OR explicitly scope: "screenshot-vs-natural-photo split is the contrast we study" |
| **Limitations transparency** | Discussion mentions limits but doesn't quantify | Add explicit "limitations" subsection with error bars / confidence intervals on all key numbers |

### 3.3 Add (missing for paper-level publication)

| Component | Why arXiv-readiness needs it | Effort |
|---|---|---|
| **Related Work §2** (formal) | Currently the LaTeX report's intro+discussion mention 12 references. arXiv version needs a proper §2 with 30+ citations across 6 subareas. The full survey is in `RELATED_WORK.md` | 1-2 days writing |
| **Contributions paragraph in §1** | Current intro is descriptive; arXiv expects 3-4 explicit bullet "we contribute …" claims | 0.5 day |
| **Limitations subsection** | Quantified, not narrative | 0.5 day |
| **Reproducibility statement** | PGD steps, eps, seeds, hardware, wall-clock — most arXiv papers in this area have this. Currently in `docs/HPC_GUIDE.md`, lift to PDF | 0.5 day |
| **Released-artifact section** | One-paragraph description of HF dataset + GitHub + HF Space — make every reader aware of the data release | 0.5 day |
| **Inter-rater reliability paragraph** (after E2) | Quantifies how much "0.227%" depends on judge subjectivity | 1 day |
| **Two new figures** | (a) Heatmap of per-(image, prompt) injection rates. (b) Histogram of LCS drift scores. Currently the report has bar chart + tables; missing one heatmap and one distribution plot | 1 day |

### 3.4 Drop (too thin or off-topic for arXiv)

| Component | Why drop |
|---|---|
| The 14-slide overall narrative scaffolding | Slides ≠ paper. Use deck content as input, not output |
| "EE141 Final Report" branding | Course branding is irrelevant in academic context |
| The triple "course project" framing | Replace with neutral academic voice — no reviewer cares this was a class assignment |
| Side commentary about HF Space demo | Demo is fine in artifact section but not paper body |
| The "v1 vs v2 evaluation methodology critique" framing | Per recent decision, just describe the current method. Don't relitigate v1's mistakes in academic prose |
| HuggingFace download counter as evidence (in body) | Mention in artifact / impact paragraph only — too soft to anchor a finding |

---

## 4. Experiments Needed

The audit's six experiments fall into two tiers. Full specs are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

### 4.1 Required for Path A (arXiv-ready, 0-2 experiments)

These are minimum to keep arXiv credibility:

- **E0a** — Fix HF dataset viewer (`StreamingRowsError` confirmed via direct fetch). Generate flat parquet manifests from existing JSON outputs. ~3-4 hours.
- **E0b** — Upload existing judge results to HF dataset (already computed locally; just `hf upload`). ~30 minutes.

Both are dataset-curation, not new science. Without them, the dataset card claim "this is the first public release of dual-dim eval scores" is literally false (scores aren't public yet).

### 4.2 Required for Path B (workshop-submission-ready, +4 experiments)

- **E1** — Direct BLIP-2 ablation (skip Stage 2, feed $x_u$ directly). Resolves the causal story. **4-6 GPU-hours**.
- **E2** — Inter-rater reliability study (re-grade 200-300 stratified pairs with a second coder; Cohen's $\kappa$). **~10 hours**.
- **E3** — Judge keyword-list sensitivity ablation (strict / baseline / loose). **3-4 hours**.
- **E4** — Systematic frontier-model transfer (top 3 cases × GPT-4o + Gemini 2.0 Flash). **~$50 API + ~5 hours**.

**Total Path B add-on**: ~25 GPU-hours + ~25 human-hours = **~3 weeks at part-time pace**.

### 4.3 Recommended but optional for Path B

- **E5** — Add 14-23 more test images (currently 7, expand to 21-30). **~30 GPU-hours**. Adds robustness across image categories — strong reviewer rebuttal but not blocking.
- **E6** — Question-set ablation (use questions 6-10 instead of 1-5 per category, compare). **~6-8 GPU-hours**. Removes the "implicit cherry-pick" risk.

### 4.4 Defer to Path C / future work

- 5-attack benchmark (typographic / steganography / cross-modal / scene spoofing) — defer to a follow-up paper
- Defense matrix — defer
- Scale study (Qwen2.5-VL-7B / LLaVA-34B / Llama-3.2-90B) — defer; a stretch goal if Path B succeeds
- Stage-1 ablations (loss function, ensemble size, training-step sweep) — defer

---

## 5. Dataset Expansion Decisions

Full analysis in [`DATASET_LANDSCAPE.md`](DATASET_LANDSCAPE.md). Summary:

### 5.1 Required before arXiv

- Fix viewer (E0a above)
- Upload judge results (E0b above)
- Rewrite dataset card structured like `JailbreakV-28K` + `wildguardmix`: Summary → Visual Teaser → Threat Model → Methodology → Schema → Mini-Leaderboard → "What this is NOT" → Reproducibility Budget → Limitations → Transferability → Citation → License
- Add 3 PNG case-study images directly in the card body
- Add inter-rater agreement number (after E2)
- Add explicit content/safety disclaimer (1 paragraph, top of card)
- Update citation block once arXiv ID is assigned

### 5.2 Strongly recommended for Path B (alongside experiments)

- **Add 2-3 more VLMs**: Phi-3.5-Vision / LLaVA-1.6 / Llama-3.2-Vision-11B. Wrappers exist in `models/registry.py` (excluded for VRAM/version reasons). Even partial coverage triples credibility.
- Promote the dual-axis methodology to a top-of-card section
- Add a "What this dataset is NOT" callout (copy `prodnull/...` pattern)
- Add transferability column once E4 is run

### 5.3 Don't expand on these axes

- More attack prompts beyond 7 — diminishing returns
- More raw response pairs without methodological additions — size for its own sake doesn't help
- v2-dev's 4 new attack categories — defer to follow-up
- Auth-gating the dataset — public CC-BY-4.0 wins, every gated peer has 10-100× fewer downloads

---

## 6. Paper Outline Summary

Full section-by-section outline in [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md). High-level structure (8-9 pages, NeurIPS-style):

```
1. Introduction                                            ~1 p
2. Related Work                                            ~1 p
3. Building Blocks (UAA + AnyAttack + dual-dim eval)       ~1.5 p
4. Experimental Setup                                      ~1 p
5. Results                                                 ~1.5 p
   5.1 Headline numbers
   5.2 Per-VLM (BLIP-2 immunity)
   5.3 Per-prompt
   5.4 Per-image
   5.5 Inter-rater agreement (NEW)
   5.6 Frontier-model transfer (NEW, after E4)
6. Case Studies                                            ~1.5 p
7. Discussion                                              ~1 p
   7.1 Why drift ≫ injection
   7.2 BLIP-2 ablation result (NEW, after E1)
   7.3 Limitations
8. Conclusion + Released Artifacts                         ~0.5 p
References                                                 ~0.5 p
Appendix A. The 60-question pool                           ~1 p
```

---

## 7. Timeline (Path A + Path B in sequence)

### Path A — arXiv preprint (3-4 weeks)

| Week | Tasks | Deliverable |
|---|---|---|
| **Week 0** (this week) | Get Tufts professor's arXiv endorsement (cs.CR primary, cs.CV/cs.LG cross-list). Send draft abstract + paper title for context. | Endorsement code |
| **Week 1** | E0a (fix HF viewer) + E0b (upload judge results). Rewrite HF dataset card per §5.1. Start §2 Related Work draft. | HF dataset card v3.0; Related Work first draft |
| **Week 2** | Convert PDF report to arXiv style: clean academic voice, drop course branding, add Contributions paragraph in §1, add Reproducibility statement. Add Released-Artifacts section. | arXiv-style PDF v0.9 |
| **Week 3** | Final pass: BibTeX cleanup, figure DPI, cross-references. Submit to arXiv. | arXiv preprint v1 (occupy citation) |

### Path B — workshop submission (additional 6-8 weeks)

| Week | Tasks | Deliverable |
|---|---|---|
| Week 4-5 | Run E1 (BLIP-2 ablation), E3 (keyword sensitivity), E2 (inter-rater partial). Write up results. | Updated §5 + §7 with new findings |
| Week 6 | Run E4 (frontier-model transfer). Add §5.6 systematic transfer subsection. | Updated transfer section |
| Week 7 | Optional: E5 (more test images) if HPC slots permit. Otherwise polish prose. | Optional dataset expansion |
| Week 8 | Push arXiv v2 with all experiments. Submit to AISec @ ACM CCS (deadline typically June-July) or AdvML Frontiers Workshop @ NeurIPS (deadline Aug-Sep). | arXiv v2 + workshop submission |
| Week 9-10 | Wait for review. Optional: prep camera-ready or rebut. | Workshop response |

### Path C — only consider if Path B accepted

Plan deferred. See `EXPERIMENTS.md` §"Top-conf experiments" for the rough scope (5 attacks × 3 defenses + scale study + mechanistic analysis).

---

## 8. Concrete Next Actions (this week)

1. **Today**: Confirm cs.CR is the right arXiv category. (See discussion in conversation history — yes, it is.)
2. **This week**: Email a Tufts professor for arXiv endorsement. Use email draft from previous turn (I can write the actual draft on request).
3. **This week**: Audit the HF dataset's broken viewer (`Error: StreamingRowsError`). Fix root cause: schema mismatch on `deepseek_vl_1_3b` field. Generate flat parquet. **3-4 hours.**
4. **This week**: Upload judge results (already on local disk under `outputs/experiments/.../results/judge_results_*.json`) to HF dataset. **30 min.**
5. **Next week**: Begin §2 Related Work draft using `RELATED_WORK.md` as input — pick the 25-30 most-cited papers for the actual references list, leave the rest in supplementary survey.

---

## 9. Risk Assessment

What could go wrong:

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| arXiv endorsement falls through (no Tufts professor responds in time) | Low | High | Have 2-3 candidate professors. Email this week. The endorsement form is one click for them. |
| BLIP-2 ablation (E1) shows the immunity is **not** Stage-2 fusion but actually gradient dilution | Medium | Low | Either result is publishable. Update §7 framing accordingly. |
| Inter-rater study (E2) shows κ < 0.6 | Low | Medium | Be transparent; discuss in §5.5 as "evaluation has subjective component, dual-axis split mitigates this" |
| GPT-4o transfer (E4) shows non-zero transfer rate | Low | Medium | Pivot the framing: "transfer is partial / signal-dependent" — still publishable, less clean story |
| Rahmatullaev / Zhang / Schlarmann publishes a stronger version concurrent with our submission | Medium | Low | Cite and differentiate. Our methodology contribution (dual-axis eval) is independent of the attack pipeline |
| Workshop deadline missed | Medium | Low | arXiv is permanent; worst case re-target next cycle |
| You don't have post-graduation bandwidth for Path B | High | Low (already have Path A done) | Path A deliverable is self-contained; Path B is genuinely optional |

---

## 10. Decision Points

You'll need to make these choices explicitly:

1. **Path A only, or commit to Path B from the start?** Recommendation: commit to Path A this week, defer Path B decision until Week 3 (after arXiv v1 is live).
2. **Workshop choice (Path B)**: AISec @ CCS vs AdvML Frontiers vs SaTML. Recommendation: AISec — strongest fit for the threat-model + dataset narrative. AdvML if AISec deadline missed.
3. **Author list**: solo? + advisor? + the professor who endorses? Recommendation: solo for v1, add advisor/professor for v2 if they contributed. arXiv supports adding authors in a v2 update.
4. **License**: code (MIT-ish), dataset (already CC-BY-4.0), paper (CC-BY-4.0 standard for arXiv). All compatible.

---

## 11. What's NOT in this plan

Out of scope:

- **A new GitHub repo** for the arXiv version — the existing `vis-inject` repo is fine. The arXiv paper can sit at `vis-inject/xarchive-report/paper/` or as a separate published artifact only on arXiv + HF.
- **Slides / talk preparation** — only relevant if accepted to a workshop with a presentation slot.
- **Funding / grant writing** — not applicable for solo work.
- **Patent / IP review** — work is being released CC-BY-4.0; nothing to file.
- **PR / blogging** — if the paper goes well, a Twitter / blog post comes after publication, not before.

---

## 12. References to supporting documents

- [`AUDIT.md`](AUDIT.md) — full audit report (1500 words, 6-dimension scorecard)
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — detailed experiment specs E0-E6 (Path B) + Path C scope
- [`RELATED_WORK.md`](RELATED_WORK.md) — 42-paper survey, organised in 6 subareas
- [`DATASET_LANDSCAPE.md`](DATASET_LANDSCAPE.md) — 30-HF-dataset survey + best practices
- [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) — section-by-section paper structure

External:
- HF dataset: <https://huggingface.co/datasets/jeffliulab/visinject>
- HF Space: <https://huggingface.co/spaces/jeffliulab/visinject>
- GitHub: <https://github.com/jeffliulab/vis-inject>
- v1.1 git tag: full course-final deliverables (PDF, slides, code zip) preserved
