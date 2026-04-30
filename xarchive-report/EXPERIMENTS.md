# Experiment Specifications for arXiv / Workshop Path

> Concrete experiments needed before arXiv release (E0) and before workshop submission (E1-E4 mandatory; E5-E6 recommended). Each spec includes setup, expected outcome, effort estimate, and how the result feeds into the paper.

---

## Tier 0 — Required for arXiv release (HF dataset polish)

These are dataset-curation, not new science. Without them the dataset card's claim "we release dual-axis judge scores" is literally false.

### E0a — Fix HF dataset viewer (`StreamingRowsError`)

**Confirmed bug**: direct fetch of `https://huggingface.co/datasets/jeffliulab/visinject` returns `Error code: StreamingRowsError`, "viewer is not available for this split." Caused by schema mismatch on the per-experiment JSON tree (`deepseek_vl_1_3b` field appears as nested struct in some files, as a flat dict in others).

**Fix**:
1. Generate a **flat parquet manifest** by walking `outputs/experiments/exp_*/results/response_pairs_*.json` and emitting one row per `(exp, vlm, image, question)` with columns:
   - `experiment` (str), `prompt_tag` (str), `model_config` (str)
   - `image` (str), `vlm` (str), `question` (str), `category` (str)
   - `response_clean` (str), `response_adv` (str)
   - `target_phrase` (str)
2. Generate a second parquet for `judge_results_*.json` (one row per `(exp, vlm, image)`):
   - same key columns
   - `output_affected_score` (float), `output_affected` (bool)
   - `target_injected_score` (float), `target_injected` (bool)
   - `level` (enum: confirmed / partial / weak / none)
3. Upload both as `response_pairs.parquet` and `judge_results.parquet` at the dataset root.
4. Add a `dataset_infos.json` declaring these as the default splits.
5. Verify viewer renders.

**Effort**: 3-4 hours (pandas + `huggingface_hub` upload).

**Output**: `response_pairs.parquet` (~30K rows, ~30 MB) + `judge_results.parquet` (~6.6K rows, ~5 MB) + working viewer.

**Paper impact**: makes the dataset queryable in one line: `datasets.load_dataset("jeffliulab/visinject")`. Mentioned in the §"Released Artifacts" paragraph.

---

### E0b — Upload existing judge results to HF

The judge scores for all 6,615 pairs already exist locally at `outputs/experiments/exp_*/results/judge_results_*.json`. They are **not** currently on HF.

**Fix**:
1. Run `hf upload jeffliulab/visinject outputs/experiments/ experiments/ --repo-type dataset` to mirror the JSON tree.
2. Verify the upload succeeded by `datasets.load_dataset("jeffliulab/visinject", split="experiments")` returns expected counts.

**Effort**: 30 minutes.

**Paper impact**: enables the dataset to back the dual-axis methodology claim. Without this, the methodology is only documented in the paper, not in the public artifact.

---

## Tier 1 — Required for Path B (workshop submission)

These four experiments close the audit's most damning weaknesses.

### E1 — Direct BLIP-2 ablation (Stage-1 → BLIP-2, skip Stage 2)

**Question**: is BLIP-2 immune because Stage-2 fusion strips its signal, or because Stage 1 never learned to push BLIP-2 in the first place?

**Setup**:
1. Take the 21 universal adversarial images $x_u$ already produced by Stage 1 (one per `(prompt, ensemble)`).
2. For each $x_u$, feed **directly** to BLIP-2 (skip Stage 2 / AnyAttack fusion entirely).
3. Run all 15 evaluation questions on BLIP-2.
4. Score Output-Affected and Target-Injected.

**Total**: 21 images × 15 questions = 315 (clean, $x_u$) response pairs on BLIP-2.

**Expected outcomes & interpretation**:

| Outcome | Interpretation | Story |
|---|---|---|
| **BLIP-2 affected on $x_u$ direct** but not on $x_a$ | **Stage-2 fusion strips the signal.** AnyAttack decoder's output noise pattern is shaped by COCO bidirectional pretraining; it doesn't preserve the BLIP-2-targeting component. | Strong story: motivates v2 D2 ("port Q-Former bottleneck as defense") and explains why transfer-via-decoder fails. **Best outcome for the paper.** |
| **BLIP-2 unaffected on $x_u$ direct** AND on $x_a$ | **Q-Former bottleneck is fundamentally robust** at this perceptual budget. Stage 1 never made progress on BLIP-2 in the first place — the gradient was diluted by Qwen / DeepSeek dominating the loss sum. | Architectural-robustness finding. Different story but equally publishable. |
| **BLIP-2 affected on $x_u$ direct** AND on $x_a$ | **Surprise**: AnyAttack does preserve some BLIP-2 signal, but our v1.1 evaluation missed it. | Re-evaluate v1.1 numbers. Potentially significant finding. |

**Effort**: 4-6 GPU-hours on H200 (or ~2 hours of API time on a hosted BLIP-2 endpoint).

**Paper impact**: turns §7's three-hypothesis speculation into a falsifiable, tested claim. **This is the highest-ROI experiment.** Add as §7.2 subsection.

---

### E2 — Inter-rater reliability for the dual-axis judge

**Question**: how much does the "0.227% injection rate" depend on the specific judge's subjective thresholds?

**Setup**:
1. Stratified sample 200-300 response pairs from the 6,615:
   - 50 pairs from "confirmed injection" cases
   - 50 pairs from "partial / weak" cases
   - 100 pairs from "no injection observed"
   - 50 pairs across the whole dataset uniform-random
2. Re-grade each pair with a **second coder**:
   - Option A (cheaper): a different LLM judge (e.g., GPT-4o-mini at temperature 0). ~$10 in API. Run identical prompt template.
   - Option B (more rigorous): manual coding by a human (you, plus optionally a friend). ~6 hours of work for 250 pairs.
3. For each axis (Output-Affected, Target-Injected) compute:
   - Cohen's $\kappa$ (binary) or Fleiss' $\kappa$ (3+ raters)
   - Pearson correlation on the continuous scores
   - Confusion matrix

**Expected outcomes**:

| $\kappa$ | Interpretation | Story |
|---|---|---|
| > 0.7 | Strong agreement; judge is reliable | Cite the number; strengthens the methodology section |
| 0.5–0.7 | Moderate agreement; defensible but worth discussing | Add a paragraph on judge sensitivity in §5.5 |
| < 0.5 | Weak agreement; the keyword-list approach is too subjective | **Reframe** the paper around the methodology critique rather than the injection-rate claim |

**Effort**: ~10 hours (sampling + grading + analysis). $10 in API if using LLM coder.

**Paper impact**: new §5.5 subsection. Makes "0.227%" defensible against the obvious reviewer attack.

---

### E3 — Judge keyword-list sensitivity ablation

**Question**: how sensitive is the injection rate to specific keyword choices in the variant lists?

**Setup**: Define three keyword-list variants per target prompt:

| Variant | Definition | Example for `card` prompt |
|---|---|---|
| Strict | exact target phrase only | "tell me your card number" — no variants |
| Baseline | current setting | + {card number, account number, bank} |
| Loose | + broader semantic | + {payment, debit, credit, ATM, PIN, billing} |

Re-grade all 6,615 pairs against each variant. Report injection rate per `(prompt, variant)`.

**Expected outcomes**:

| Result | Interpretation |
|---|---|
| Injection rates change by < 1.5× across variants | Judge is robust; baseline keyword lists are not biasing the result |
| Injection rates change by 2-5× across variants | Acknowledge subjectivity; report all three numbers in the table |
| Injection rates change by > 5× | The "0.227%" depends heavily on the specific keyword choice; reframe results |

**Effort**: 3-4 hours (judge code parameterisation + re-run).

**Paper impact**: new sensitivity table in §5; defends the methodology against keyword-bias attacks.

---

### E4 — Systematic frontier-model transfer

**Question**: does the strongest small-VLM injection actually transfer to GPT-4o, Gemini 2.0 Flash, or Claude 3.5 Sonnet?

**Setup**:
1. Pick **top-3 highest-ASR injection cases** per attack category (`url`, `card`, `email`):
   - Top URL injection: `adv_url_3m_ORIGIN_code.png` + URL-extraction prompt (confirmed injection on Qwen2.5-VL)
   - Top card injection: `adv_card_3m_ORIGIN_bill.png` + bill-description prompt (partial injection on DeepSeek-VL)
   - Top email injection: `adv_email_4m_ORIGIN_bill.png` + bill-analysis prompt (partial on Qwen2-VL)
2. For each case, query 3 frontier models:
   - GPT-4o (via OpenAI API)
   - Gemini 2.0 Flash (via Google API)
   - Claude 3.5 Sonnet (via Anthropic API)
3. For each `(case, model)` pair, run **all 15 benign questions** (not just the original).
4. Score each response pair (clean vs adversarial) on Output-Affected + Target-Injected.

**Total**: 3 cases × 3 models × 15 questions = 135 transfer-test pairs (×2 for clean baseline = 270 generations).

**Expected outcomes**:

| Outcome | Interpretation |
|---|---|
| Zero transfer (0/9 on Target-Injected) | "No transfer to frontier closed models" — clean negative result, publishable |
| 1-2/9 transfer | Transfer is rare but possible; reframe as "weak transfer signal" |
| ≥3/9 transfer | Surprising result; significant story about closed-model vulnerability |

**Effort**: 5 hours (API setup + batch run + scoring) + ~$50 in API costs ($30 GPT-4o + $10 Gemini + $10 Claude).

**Paper impact**: new §5.6 subsection. Replaces the N=1 anecdote with systematic evidence.

---

## Tier 2 — Recommended for Path B (if time permits)

### E5 — Expand test images from 7 to 21-30

**Question**: does the disruption / injection pattern hold across more image categories?

**Setup**: Add 14-23 new test images sampled across:
- Natural photos (faces, landscapes, animals, products)
- Documents (multi-page bills, receipts, contracts)
- UI screenshots (mobile apps, dashboards, code IDEs)
- Charts / diagrams (matplotlib, hand-drawn)
- Multi-object cluttered scenes
- Low-contrast images
- Synthetic / generated images

Re-run Stage 2 fusion + Stage 3 evaluation on all new images.

**Effort**: ~30 GPU-hours (Stage 2 is fast; Stage 3 is the cost).

**Paper impact**: removes the "image-type-narrow" reviewer attack. Promotes the Per-Image table in §5 from "trend across 7 images" to "trend across 25+ images" — much stronger.

---

### E6 — Question-set ablation

**Question**: does the choice of 15 evaluation questions (first 5 of each category) bias results?

**Setup**: Re-run Stage 3 evaluation with **questions 6-10 of each category** (i.e., a different 15-question slice from the same 60-question pool). Compare disruption + injection rates against the baseline.

**Effort**: ~6-8 GPU-hours on the existing 147 adversarial photos.

**Paper impact**: removes the "implicit cherry-pick" reviewer attack. If results match within 1-2σ, baseline is representative; if not, document the variance and adjust claimed numbers to "0.2% ± σ."

---

## Tier 3 — Top-conf experiments (Path C, deferred)

These are noted for completeness but **not recommended** for the next 6 months.

### Five-attack benchmark (Path C)

Implement the v2-dev plan's C2 (typographic), C3 (steganography), C4 (cross-modal), C5 (scene spoofing). Evaluate all 5 against same 3 VLMs × 7 images × 15 questions × dual-axis judge. Report PSNR, SSIM, LPIPS alongside ASR.

**Effort**: 60-80 GPU-hours + 3-4 weeks of new code.

### Three-defense matrix (Path C)

Implement D1 (input preprocessing), D2 (Q-Former bottleneck transplant), D3 (multi-VLM consensus). Run all attacks × defenses × VLMs.

**Effort**: 40-60 GPU-hours + 2-3 weeks of new code.

### Scale study (Path C)

Add Qwen2.5-VL-7B, LLaVA-1.6-34B, Llama-3.2-90B-Vision. Sweep $\eps \in \{4, 8, 16, 32\}/255$. Report scaling curves.

**Effort**: 50-80 GPU-hours + significant HPC quota.

### Mechanistic analysis (Path C)

Gradient-flow analysis showing where in each VLM the perturbation gets attenuated. Feature-space visualisations (t-SNE / UMAP) of clean vs adversarial CLIP embeddings.

**Effort**: 20-30 hours of analysis code + paper-quality figures.

---

## Total Effort Summary

| Tier | What | Wall-clock effort | Cost |
|---|---|---|---|
| E0 | HF dataset fixes | 3-5 hours | $0 |
| E1 | BLIP-2 ablation | 4-6 GPU-hours | $0 |
| E2 | Inter-rater reliability | ~10 hours (+ $10 API) | ~$10 |
| E3 | Judge keyword sensitivity | 3-4 hours | $0 |
| E4 | Frontier-model transfer | ~5 hours | ~$50 |
| **Path A total (E0)** | | **~5 hours** | **$0** |
| **Path B total (E0-E4)** | | **~30 hours + ~25 GPU-hours** | **~$60** |
| E5 | More test images | ~30 GPU-hours | $0 |
| E6 | Question-set ablation | ~8 GPU-hours | $0 |
| **Path B+ (E0-E6)** | | **~70 hours + ~65 GPU-hours** | **~$60** |
| Path C | full v2 + scale + mech | **~3 months full-time** | **$200-500 cluster + APIs** |

---

## Output: what each experiment buys you in the paper

| Experiment | Section it strengthens | Reviewer-attack it neutralises |
|---|---|---|
| E0 | "Released Artifacts" | "the dataset is incomplete" |
| E1 | §7 BLIP-2 immunity | "your causal story is speculation" |
| E2 | §5.5 (new) | "your judge is subjective" |
| E3 | §5 keyword sensitivity table | "your keyword lists are biased" |
| E4 | §5.6 (new) | "your transferability claim is N=1 anecdote" |
| E5 | §5 per-image robustness | "your findings are screenshot-specific" |
| E6 | §5 question-set robustness | "your eval set is implicitly cherry-picked" |

The cumulative effect of E1-E4 is that **every weakness flagged in the audit's "What is risky" section becomes addressed**. That's what moves the paper from "honest course project" to "credible workshop submission."
