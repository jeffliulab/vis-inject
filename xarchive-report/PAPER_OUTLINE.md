# arXiv Paper — Section-by-Section Outline

> Target: 8-9 pages (NeurIPS-style format) + appendix + references.
> Reuse the existing LaTeX project at `report/pdf/` as the starting point — drop course branding, add Related Work, strengthen Contributions.

---

## Title (working)

> **VisInject: Disruption ≠ Injection — A Dual-Dimension Evaluation of Universal Adversarial Attacks on Vision-Language Models**

Alternative titles (less catchy but more explicit):

- "How Often Does an Imperceptible Adversarial Image Actually Plant Attacker-Chosen Content in a VLM Response? Disruption Is Broad, Payload Delivery Is Rare"
- "On Measuring Visual Prompt Injection: A Dual-Axis Benchmark for VLM Adversarial Attacks"

The first title leads with the methodological contribution (drift ≠ injection); reviewers will read the title's claim and the abstract. The current course PDF's title ("VisInject: Adversarial Prompt Injection on Vision-Language Models") is a fine *project* name but lacks the contribution claim.

---

## Abstract (~200 words)

**Structure** (4 sentences each ~50 words):

1. **Motivation**: Recent universal-attack-on-VLM papers report attack success rates in the 50-80% range, suggesting the visual modality is highly vulnerable to imperceptible adversarial perturbations as a prompt-injection vector.
2. **Gap**: Reported "ASR" conflates two distinct events — (i) the model's output was perturbed, (ii) the attacker's chosen target string actually appeared. We argue these are different things and existing benchmarks have no clean way to separate them.
3. **Method + result**: We compose Universal Adversarial Attack [Rahmatullaev 2025] with AnyAttack [Zhang CVPR 2025] under an L∞ = 16/255 budget, run a 6,615-pair systematic sweep across 4 open VLMs × 7 attack prompts × 7 test images, and introduce a programmatic dual-axis evaluation (Output-Affected via LCS drift; Target-Injected via keyword/regex). Headline: 66% Output-Affected coexists with 0.227% Target-Injected — a 290× gap.
4. **Implication**: Closer inspection reveals BLIP-2's Q-Former bottleneck is fully immune (0/2,205) even when included as a Stage-1 surrogate; literal injection only succeeds on screenshots whose semantics already invite text transcription. We release 6,615 (clean, adversarial) response pairs with dual-axis judge scores at huggingface.co/datasets/jeffliulab/visinject.

---

## §1 Introduction (~1 page)

**Open with the puzzle**: "If small open VLMs can be jailbroken by imperceptible perturbations 60-80% of the time (per recent literature), why do practical adversarial-image attacks rarely lead to credential theft, URL injection, or misinformation in the wild?"

**Lead claim**: Reported ASR conflates *output drift* with *payload delivery*. These are different.

**Contributions** (4-bullet block):

- **C1.** A dual-dimension evaluation framework that separates Output-Affected (drift, by LCS overlap) from Target-Injected (payload, by keyword/regex). Programmatic, deterministic, ~5 minutes per file on CPU.
- **C2.** A 6,615-pair systematic sweep showing the two axes diverge by 290× on the same data — disruption is broad, injection is rare and clustered.
- **C3.** An architecture finding: BLIP-2's Q-Former bottleneck is fully immune (0/2,205) even when used as a Stage-1 surrogate. We provide an ablation locating the immunity in [Stage-2 fusion / Q-Former bottleneck — pick after E1 runs].
- **C4.** A public dataset (HF) bundling 21 universal images, 147 adversarial photos, and the dual-axis scores per pair.

**Roadmap paragraph** (1-line summaries of §2-§7).

---

## §2 Related Work (~1 page)

Six paragraphs, one per subarea. Each cites 4-6 papers from `RELATED_WORK.md`. Use ~30 references total.

1. **Universal adversarial attacks on (M)LLMs** — Rahmatullaev 2025 (the direct reuse for our Stage 1), Schlarmann ICCV 2023 (closest threat model), Qi AAAI 2024 (visual jailbreak), Carlini NeurIPS 2023 (multimodal weak link). Differentiate: we use universal attack as the *signal* not the *attack*; the attack is the composition with AnyAttack and the dual-dim evaluation.
2. **Indirect prompt injection** — Greshake AISec 2023 (foundational), Bagdasaryan 2023 (closest spirit), Liu USENIX 2024 (formalization), Yi BIPIA KDD 2025 (text-side benchmark). Differentiate: we are the visual-modality analogue, and we measure delivery not just disruption.
3. **Multimodal jailbreaks** — Zou 2023 GCG (text analogue), HADES ECCV 2024 (visual jailbreak SOTA), FigStep AAAI 2025 (typographic), Image Hijacks ICML 2024 (closest behaviour-matching analogue). Differentiate: those attack safety alignment; we attack output integrity. Different threat model.
4. **VLM adversarial robustness** — Schlarmann ICCV 2023 + Robust CLIP ICML 2024 (defense-side prior art), Qi AAAI 2024 (visual jailbreak), survey [Liu 2025]. Cite as background; no direct comparison since we're attack-side.
5. **Defenses** (1 paragraph) — VLGuard ICML 2024, Robust CLIP ICML 2024, ECSO ECCV 2024 (the most likely to neutralise our attack — we discuss in §7), DiffPure ICML 2022 (purification baseline). Acknowledge but don't engineer-around.
6. **Evaluation methodology / benchmarks** — HarmBench ICML 2024, JailbreakBench NeurIPS D&B 2024, MM-SafetyBench ECCV 2024, MMJ-Bench AAAI 2025, LLM-as-Judge NeurIPS D&B 2023. Differentiate: those evaluate adversarial-prompt × adversarial-image; we evaluate *benign-prompt × adversarial-image*. Different threat surface.

---

## §3 Building Blocks (~1.5 pages)

Reuse the existing PDF §3 with light edits:

**§3.1 Stage 1 — UAA** (Rahmatullaev 2025): four named techniques — image reparameterisation, masked cross-entropy loss, multi-prompt training, multi-model ensemble loss. We enable quantization-noise robustness; ablate off multi-answer / localisation / Gaussian blur. Cite paper 1 explicitly. ~0.5 page.

**§3.2 Stage 2 — AnyAttack** (Zhang CVPR 2025): self-supervised pretrained CLIP-encoder + decoder produces ε-bounded noise. We reuse public `coco_bi.pt` weights without retraining. ~0.5 page.

**§3.3 Stage 3 — Dual-axis evaluation (our contribution)**: definitions of Output-Affected (LCS drift, threshold > 0) and Target-Injected (keyword/regex match against curated variants per prompt). Cite Liu USENIX 2024 for "literal target appears" definition. Cite LLM-as-Judge NeurIPS 2023 to motivate why we *don't* use a judge model. Note that both checks are deterministic and programmatic (~5 min per file on CPU). **Add an explicit list of variant keywords per prompt** (visible audit trail). ~0.5 page.

(Composition figure stays as Figure 1.)

---

## §4 Experimental Setup (~1 page)

**Mostly reuse existing §4** with small updates:

- 7 target prompts (table). Add motivation for split between literal-text and semantic-class.
- 7 test images figure (existing 7-image grid). Add 1-line on each (natural / screenshot type).
- 4 VLMs (table) + 3 ensembles. **If E1 runs**: also note BLIP-2 ablation setup (direct $x_u → $ BLIP-2).
- 60-question pool (point at Appendix A) — emphasise the 3-category split mapping back to threat-model scenarios.
- Compute: H200 80GB, 7-19 min per Stage-1 run, ~30 min per Stage-3 (image, experiment) row.
- **Reproducibility statement**: PGD steps = 2000, eps = 16/255, lr = 1e-2, ensemble seeds (table).

---

## §5 Results (~1.5 pages)

**§5.1 Aggregate** — Two big numbers: 66% Output-Affected, 0.227% Target-Injected, 290× gap. Cite the gap as the central empirical finding.

**§5.2 Per-VLM** — Bar chart (existing Figure 3) + table. **Make BLIP-2 immunity the lead** — 0/2,205 vs 8.45/10 on Qwen2.5-VL.

**§5.3 Per-prompt** — Existing table. Note that disruption is flat (~66% across all 7 prompts) but injection varies by 5× across prompts.

**§5.4 Per-image** — Existing table. Note injection clusters on screenshots (`code`, `bill`, `cat-collage`).

**§5.5 (NEW) Inter-rater reliability** — After E2 runs. Report Cohen's $\kappa$ on a 200-300-pair stratified sample. Goal: $\kappa$ > 0.7 to defend the methodology.

**§5.6 (NEW) Frontier-model transfer** — After E4 runs. 3 cases × 3 closed models = 9 transfer tests, each with 15 questions. Report per-case-per-model Output-Affected + Target-Injected. Goal: clear "transfer is rare" claim.

(Promote E3 keyword-sensitivity to a small sub-table in §5.3.)

---

## §6 Case Studies (~1.5 pages)

**Reuse the two existing case studies** but with academic prose:

- **Case A: URL injection on a code screenshot (confirmed)** — `exp_url_3m`, Qwen2.5-VL-3B. Show clean image / adv image / clean response excerpt / adv response excerpt with `http://www.example.com` highlighted. Why this lands: code screenshots prime the model to transcribe text.
- **Case B: Payment-info injection on a bill (partial)** — `exp_card_3m`, DeepSeek-VL-1.3B. Show same layout. Why this is partial: AnyAttack fusion preserves the *semantic class* (payment vocabulary) but drops the literal target ("card number"); model hallucinates "account number" instead.

**Add a third case (NEW)**: `exp_news_2m`, Qwen2.5-VL-3B on `cat` image. Weak fragment injection — politically-charged words appear as fragments without coherent meaning. Useful to show what "weak" injection looks like.

---

## §7 Discussion (~1 page)

**§7.1 Why drift ≫ injection** — Three reasons:
1. **Decoder-fusion erases payload specifics**: AnyAttack decoder pretrained on COCO bi-directional preserves transformer-style adversarial signal but doesn't preserve target-specific tokens.
2. **Target VLM needs a semantic invitation**: response space dominated by "a dog" leaves no room for a URL; response space of a code screenshot already contains URL-shaped tokens.
3. **Architecture wins over parameter count**: BLIP-2 immune (Q-Former bottleneck); discussed in §7.2.

**§7.2 BLIP-2 immunity ablation (NEW after E1)** — Direct $x_u →$ BLIP-2 result. Either confirms Stage-2 fusion is the cause (publishable mechanism) or confirms architectural robustness (publishable finding). Discuss implications for VLM design.

**§7.3 Limitations** — Explicit:
- White-box ensemble small (4 models, all ≤3B parameters)
- Keyword-judge keyword lists are hand-curated (mitigated by E3)
- Transfer test has 9 (case, model) cells (mitigated by E4 vs the original N=1)
- 7 test images, all natural-or-screenshot (suggested for v2)

---

## §8 Conclusion + Released Artifacts (~0.5 page)

**Conclusion** (3 sentences):

> We composed two existing universal-attack methods with a new dual-axis evaluation and ran a 6,615-pair sweep on 4 open VLMs. The 290× gap between Output-Affected and Target-Injected suggests existing universal-attack ASR claims have been over-stating *delivery* by measuring *drift*. We release dataset and code; the next step is a systematic five-attack × three-defense study (deferred to follow-up).

**Released artifacts**:

- HuggingFace dataset: `huggingface.co/datasets/jeffliulab/visinject` — 21 universal images + 147 adv photos + 6,615 response pairs + dual-axis judge scores. CC-BY-4.0.
- HuggingFace Space: `huggingface.co/spaces/jeffliulab/visinject` — interactive demo.
- GitHub: `github.com/jeffliulab/vis-inject` — full source code, MIT license. v1.1 git tag preserves the exact code used in this paper.
- Companion paper preprint: arXiv:[insert ID].

---

## §References (~0.5 page)

~30 references, drawn from `RELATED_WORK.md`. BibTeX cleanup pass needed (cross-ref author orderings via crossref.org).

---

## Appendix A — The 60-question pool (~1 page)

Reuse existing PDF Appendix A: full enumeration of 60 questions in 3 categories, with mapping to threat-model scenarios.

---

## Optional Appendices

If page budget allows:

- **Appendix B — Full keyword variant lists per target prompt** (transparency for the Target-Injected check; addresses E3's auditability concern)
- **Appendix C — All 12 case-study response panels** (full transcripts of each confirmed/partial/weak case from `outputs/succeed_injection_examples/`)
- **Appendix D — Per-VLM per-prompt × per-image heatmap** (NEW — see PLAN.md §3.3 about adding figures)
- **Appendix E — Hardware / wall-clock budget per experiment row** (reproducibility)

---

## Length Budget Check

| Section | Target pages |
|---|---|
| §1 Introduction | 1.0 |
| §2 Related Work | 1.0 |
| §3 Building Blocks | 1.5 |
| §4 Experimental Setup | 1.0 |
| §5 Results | 1.5 |
| §6 Case Studies | 1.5 |
| §7 Discussion | 1.0 |
| §8 Conclusion + Artifacts | 0.5 |
| References | 0.5 |
| Appendix A (questions) | 1.0 |
| **Main body total** | **9.5 pages** |
| Optional appendices | +2-3 pages |

NeurIPS / ICML main-track papers typically cap at 9 pages of body + unlimited references / appendix. arXiv has no page limit. Workshop venues vary (4-8 pp). Budget aligns.

---

## Style Notes

- **Drop the course-project framing entirely**. No "EE141", no "course final report", no "we built this in 3 weeks." Voice is third-person academic.
- **Avoid first-person plural for results** ("we find") in favour of passive ("the analysis reveals", "the sweep shows") where appropriate, but consistent first-person plural elsewhere is fine for a single-author or co-authored paper.
- **Use $\eps = 16/255$ consistently** with other notation from the field (some papers use $0.0625$, some use $16/255$ — pick one and stick).
- **Use \citep / \citet correctly** with natbib. Sentence-internal citations should be \citep; subject citations should be \citet.
- **Drop emphasis-via-bold inside running prose**. Use \emph or italics. Bold is for table cells only.
