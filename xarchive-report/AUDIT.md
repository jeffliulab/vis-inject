# VisInject v1.1 — Honest Audit for arXiv Readiness

> Source: parallel audit by an Explore subagent that read all source files + the v2-dev branch's planned future work.
> Date: 2026-04-30.
> Tone: candid. No diplomatic softening.

---

## Overall Assessment

VisInject v1.1 is a **technically solid but empirically underwhelming** project that sits at a critical crossroads. The dual-dimensional evaluation methodology is a genuine contribution that cleanly separates disruption from injection — this will be cited. However, the core empirical claim (0.227% injection rate across 6,615 pairs) is so thin that it reads more as a negative result than a positive finding. The project's honesty about this gap is admirable but creates arXiv positioning challenges.

The work is reproducible, the code is clean, the limitations are acknowledged. For a top-tier venue submission, substantial new evidence is needed; as a workshop paper or arXiv preprint with a focused negative result, it's publishable now. The authors have a clear v2 roadmap (5 attacks × 3 defenses × 2 frontier models), but it exists only on `v2-dev` as unexecuted promises.

---

## Dimension Scorecard

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| **1. Methodology Rigor** | 3.5 / 5 | Dual-dim evaluation (affected vs. injected) is clean and addresses the core problem. Output-Affected via LCS drift is sound. Target-Injected via keyword/regex relies on hand-curated lists (card → {card number, account number, bank}) without principled justification for inclusion / exclusion. A reviewer could plausibly argue "why include 'account number' as a card variant, but not 'debit card'?" The lists are defensible but not bulletproof. **Biggest weakness**: no inter-rater reliability study, no ablation on judge thresholds. |
| **2. Claim Strength** | 2.5 / 5 | "66% disruption" is solid and well-replicated (consistent across 3 transformer-style VLMs, 7 prompts, 7 images). "0.227% injection rate" is defensible as a negative result but **thin as a positive finding** — only 15 cases in 6,615 pairs, of which 2 are "confirmed" and 5 are "weak". Claim "BLIP-2 fully immune (0/2,205)" is surprising and well-established empirically, but the **causal explanation** (Q-Former bottleneck) is speculative without ablation. "Architecture matters more than size" is fair but not surprising in VLM literature. "Does not transfer to GPT-4o" is undercut by N=1 manual test. |
| **3. Reproducibility** | 4 / 5 | Code structure is clean: unified `config.py`, externalized attack parameters, clearly named wrappers, staged pipeline. Dataset with 12 curated cases is on HuggingFace. The 60-question pool is in `dataset.py` with deterministic ordering. Judge code is deterministic (no randomness, no API calls). **One weakness**: HPC-specific paths (Tufts cluster) make re-running Stage 3a on a fresh clone non-trivial. Stage 1 and 2 are locally reproducible; Stage 3a requires either HPC access or 15+ hours on a single GPU. |
| **4. Experimental Scale** | 2.5 / 5 | Design is **comprehensive in breadth but shallow in depth**. 4 white-box VLMs (all ≤3B params, all transformer-style except BLIP-2), 7 attack prompts, 7 test images, 15 eval questions per pair = 6,615 pairs total. Compared to related work: Carlini et al.'s adversarial-patch work used 10K images; Qi et al. used 20+ target objects; HarmBench includes 33 LLMs/defenses. VisInject's 7 images are a reasonable pedagogical subset, but "only tested on screenshot-heavy scenarios" is a real limitation. N=1 transfer test to GPT-4o is well below the "systematic" threshold. **For a workshop**: adequate. **For ICCV / NeurIPS**: needs 50+ images, 2-3 frontier-model transfers, and ablations. |
| **5. Limitation Transparency** | 4 / 5 | §7 (Discussion → Limitations) is honest: acknowledges small white-box ensemble, keyword-matching softness, and single GPT-4o case. Authors explicitly state the BLIP-2 immunity is "surprising" and could be Stage-2 fusion or gradient dilution. **One honest omission**: the evaluation set (first 5 questions per category) is deterministic, creating risk of implicit overfitting to those specific questions in the judge's keyword lists. Also missing: power analysis — is 6,615 pairs enough to detect small effects reliably? |
| **6. Surprise Factor** | 3 / 5 | **Most surprising**: BLIP-2 immunity despite being a Stage-1 surrogate (0% across 2,205 pairs). This is a genuine empirical contribution and raises research questions about Q-Former robustness. **Expected**: 66% disruption on transformer-style VLMs at $\eps = 16/255$ (aligns with other adversarial-attack literature). **Disappointing but honest**: injection rate of 0.2% is far lower than disruption rate, suggesting AnyAttack fusion erases payload specificity. Presented as a negative finding, not a research direction. The gap itself (disruption ≠ injection) is the core story and is well-told. **No shock-value claim present**. |

---

## What is Solid (Keep for arXiv)

1. **Dual-dimensional evaluation framework** — The cleanest contribution. Separating Output-Affected (LCS drift) from Target-Injected (keyword match) is methodologically sound and exposes the disruption-vs-injection gap that earlier judge-based work had hidden. **This framework is reusable**: any future VLM-attack paper should report both dimensions.

2. **BLIP-2 immunity finding** — Across 2,205 evaluation pairs, BLIP-2 shows 0% affected / 0% injected while Qwen / DeepSeek show 98–100% affected. The three competing explanations (Stage-2 fusion strips signal / resolution bottleneck / gradient dilution) are plausible and testable. **This is a surprising result that deserves publication.**

3. **Reproducible pipeline** — v1.1 tag is locked, `config.py` is externalized, dataset is on HuggingFace (300 downloads / month suggests uptake). Code structure is modular; a careful reader can reproduce Stage 1 & 2 locally in ~4 hours. Stage 3 is HPC-dependent but the judge is deterministic and local.

4. **Thoughtful threat model** — Three operational scenarios (user upload, agent screenshot reading, tool-replay channels) are realistic and well-motivated. The 60-question pool covering user / agent / screenshot categories is a deliberate design to avoid prompt-distribution shift.

5. **Honest case studies** — The manifest documents 12 cases (2 confirmed, 3 partial, 5 weak) with side-by-side responses. Instead of cherry-picking wins, the authors show the full spectrum and explain *why* injection only lands on screenshots whose semantics already invite text transcription.

---

## What is Thin (De-emphasise or Strengthen)

1. **0.227% injection rate** — Reporting this as a primary claim is risky. It reads as "we tried a thing, it mostly didn't work," which is honest but not compelling for a top venue. **Reframe**: lead with BLIP-2 immunity + disruption rate; present injection as "a case study showing why transfer is hard" rather than "our main result." For arXiv, emphasise the negative result's methodological value: "we show how to measure payload delivery correctly, unlike prior work."

2. **Single GPT-4o transfer test** — N=1 is anecdotal. The reported result (GPT-4o correctly identifies noise, doesn't inject) is suggestive but not generalisable. **Strengthen**: either (a) systematically test top-3 injection cases on ChatGPT + Gemini (E4 from `EXPERIMENTS.md`), or (b) explicitly frame the single test as "no evidence of transfer, not proof of no transfer."

3. **Keyword/regex judge for Target-Injected** — Hand-curated variant lists (card → {card number, account number, bank}) lack principled justification. A reviewer could argue the lists are biased toward the observed cases. **Strengthen**: ablate the judge — report injection rates with loose vs. strict keyword matching; document the rationale for each variant.

4. **7 images is limited** — All test images are either natural photos (3) or screenshot-like (4). No: highly complex scenes, low-contrast images, images with overlay text, synthetic diagrams. Creates impression that findings may be image-type-specific. **Strengthen for v2**: add 3-5 more image types to show robustness.

5. **"Architecture matters more than size" claim** — Comparing 3B (Qwen2.5) vs 2.7B (BLIP-2) is not a fair size comparison; the architecture difference (direct cross-attention vs Q-Former bottleneck) is **confounded** with size. **Reframe**: "Q-Former architecture provides robustness" rather than "architecture beats size."

---

## What is Risky (Review Will Attack Here)

1. **Causal explanation for BLIP-2 immunity is speculative** — Three competing hypotheses are listed but no direct ablation. A reviewer will ask: "Did you feed the universal image $x_u$ directly to BLIP-2 without Stage 2, bypassing the Decoder?" This is `EXPERIMENTS.md` E1. Without it, the immunity story is observational, not causal.

2. **Keyword-list audit trail is weak** — The judge's target-phrase variants are listed in code comments but lack provenance. For example, `card` matches {card number, account number, bank}. Why not credit-card? Why not PIN? A reviewer could rerun the evaluation with slightly different keyword lists and claim the injection rate is actually 0.5%. **Risk mitigation**: release the judge code and keyword lists with the arXiv paper; invite reproducibility.

3. **No inter-judge or inter-rater reliability study** — v1 evaluation used 3 LLM judges; v2 switched to single human (Claude agent). What if a different human coder rates the same 6,615 pairs? What is Cohen's $\kappa$ or Fleiss $\kappa$? A reviewer could argue the "0.2% injection" is an artefact of the judge's subjective thresholds.

4. **Stage 3 eval set (first 5 questions per category) is implicit cherry-pick** — Authors say they picked 15 questions to keep evaluation tractable. But what if a different 15 questions yield different injection rates? No ablation study on question-set choice.

5. **Transfer study is a single anecdote** — "GPT-4o correctly identifies noise, rejects injection on the URL-to-code case" is not a claim of "no transfer generally." Reviewer will ask for: (a) multiple frontier models, (b) multiple attack cases, (c) systematic transfer to other open models.

6. **HPC reproducibility** — Stage 3a (response-pair generation) requires GPU + 15-30 hours per model. A researcher on a single consumer GPU (RTX 4090, 24GB) cannot reproduce all 6,615 pairs in a weekend. **Risk**: reviewer claims the work is not "practically reproducible" outside a cluster environment. **Response**: Stage 1 & 2 are locally reproducible; Stage 3a can be approximated with a cached universal image.

---

## Confidence Levels (per claim)

- **BLIP-2 immunity (2,205 pairs, 0% affected)**: **Very high confidence**. Clear empirical result, well-documented, surprising enough to warrant publication.
- **Disruption rate (66% across 6,615 pairs)**: **High confidence**. Consistent across models, prompts, images.
- **Injection rate (0.227%)**: **Medium confidence**. Small absolute count (15 cases), keyword-judge sensitivity unchecked, question-set choice not ablated. Likely true value is in range 0.1%–0.5% depending on judge strictness.
- **No transfer to GPT-4o**: **Low confidence**. Single manual test. True transfer rate likely in range 0%–20% given larger model's robustness.

---

## Closing Recommendation

**Publish on arXiv now (Path A).** The work is honest, reproducible, and the dual-dimensional evaluation is a genuine methodological contribution. The BLIP-2 finding is surprising and well-documented. The 0.227% injection rate is a **negative result, but negative results are science**: the paper's value lies in showing how to measure the gap between disruption and injection correctly, and in documenting why payload delivery is so much harder than output perturbation.

**For a workshop submission (Path B, +6-8 weeks):** complete experiments E1–E4 from `EXPERIMENTS.md`. The five-attack benchmark and three-defense matrix from v2-dev are natural extensions but should be saved for a follow-up paper.

**For a top-conf push (Path C, 6-12 months):** all of Path B plus E5, E6, plus the v2-dev unexecuted experiments plus a scale study plus mechanistic analysis. **Not recommended right now** — work is too thin without a major investment.

Do not oversell the 0.227% as a success. Lead with BLIP-2 robustness and methodology, position injection as a case study in why transfer is hard.
