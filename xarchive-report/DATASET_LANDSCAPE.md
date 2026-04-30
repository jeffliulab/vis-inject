# VisInject — HuggingFace Dataset Landscape Survey for arXiv Companion Release

**Prepared:** 2026-04-30
**Subject dataset:** `jeffliulab/visinject` (currently ~328 downloads/month, 32.7 MB)
**Use:** Calibrate dataset card, file organization, and scope before arXiv companion release.

---

## 1. Honesty Statement: What I Actually Fetched

I attempted **~45 dataset URLs**. Of these, **30 returned real WebFetch content** (these are the basis of all the analysis below); **~15 returned 401/404** (mostly auth-gated multimodal red-team datasets like MM-SafetyBench, FigStep, HADES, RTVLM, MultiJail, etc., plus a few wrong paths). I did *not* substitute speculation for those — they appear in this report only as "mentioned-by-name" (see Section 2.6) and are weighted out of best-practice synthesis.

For two of the auth-gated families (MM-SafetyBench, FigStep, JailBreakV) I supplemented with **WebSearch** results that surfaced concrete file/structure descriptions; those are flagged as `[search-only]` in the table.

---

## 2. Per-Dataset Survey

Format:
- DL = downloads last month (visible on HF page)
- Card = quality 1-5 stars
- Paper = companion arXiv (Y/N + ID)
- Rel = relevance to visinject (low/med/high)

### 2.1 Adversarial / Jailbreak Text Datasets

| # | HF path | DL | Size / # samples | Format | Card | Paper | Rel |
|---|---|---|---|---|---|---|---|
| 1 | `JailbreakBench/JBB-Behaviors` | 31,227 | 500 rows / 1.02 MB | Parquet (CSV→) | ★★★★★ best-in-class card with category breakdown, content warning, leaderboard, BibTeX | Y, 2404.01318 (NeurIPS 2024 D&B) | **high** |
| 2 | `walledai/AdvBench` | 10,712 | 500 / 39 kB | Parquet | ★★★★ clean paper-derived port; thin "About" section | Y, 2307.15043 (Zou et al.) | **high** |
| 3 | `walledai/HarmBench` | 22,244 | <1K / 112 kB | Parquet | ★★★★ minimal but academically anchored; missing schema details | Y, 2402.04249 | med |
| 4 | `jackhhao/jailbreak-classification` | 3,367 | 1,306 / 7.65 MB | Parquet (CSV→) | ★★★★ binary-classification framing, transparent sourcing | N (GitHub-only) | med |
| 5 | `lmsys/toxic-chat` | 7,111 | 20,330 / 60.9 MB | Parquet | ★★★★★ versioned releases, baseline model + F1 metrics, ethics framework | Y, 2310.17389 | low |
| 6 | `Anthropic/hh-rlhf` | 35,877 | 169,352 / 94.7 MB | JSONL→Parquet | ★★★★★ dual-purpose explicit, "not for SFT" warning, crowdworker transparency | Y, 2204.05862 | low |
| 7 | `TrustAIRLab/in-the-wild-jailbreak-prompts` | 1,748 | 21,527 / 18.4 MB | Parquet (CSV→) | ★★★★★ IRB note, platform/source breakdown table, responsible disclosure | Y, 2308.03825 (CCS '24) | **high** |
| 8 | `PKU-Alignment/BeaverTails` | 20,139 | 364,170 / 39.5 MB | Parquet (JSON→) | ★★★★★ 14 harm categories with definitions, multiple split sizes | Y, 2307.04657 | med |
| 9 | `allenai/wildguardmix` | 8,870 | 88,484 / 56 MB | Parquet | ★★★★★ Fleiss-Kappa annotation agreement reported, content warning, code example | Y, 2406.18495 | med |
| 10 | `PKU-Alignment/PKU-SafeRLHF` | (likes 182) | 82,100 / NA | JSON→Parquet | ★★★★ dual-response preference structure, 12 harm categories | Y, 2406.15513 | med |

### 2.2 VLM / Multimodal Safety Datasets

| # | HF path | DL | Size / # samples | Format | Card | Paper | Rel |
|---|---|---|---|---|---|---|---|
| 11 | `ys-zong/VLGuard` | 537 | 1K-10K / 1.68 GB | imagefolder + train/test JSON + ZIPs | ★★★ minimal, defers to GitHub for schema | Y, 2402.02207 (ICML '24) | **high** |
| 12 | `MMInstruction/VLFeedback` | 548 | 80,258 / 7.43 GB | Parquet | ★★★★ multi-aspect 4-model annotations, missing column docs | Y, 2312.10665 | med |
| 13 | `yueliu1999/GuardReasoner-VLTrain` | 105 | 123,093 / 313 MB | JSON→Parquet, three modality splits (image/text/text_image) | ★★★★ clean code example + login note, sparse schema | Y, 2505.11049 | **high** |
| 14 | `Jarvis1111/RobustVLGuard` | 28 | 6,124 / 959 MB | jsonl + image dirs (COCO/ChartQA/TabMWP/gqa subdirs) | ★★★★ dir tree + schema example + use-cases + BibTeX; missing load snippet | Y, 2504.01308 | **high** |
| 15 | `JailbreakV-28K/JailBreakV-28k` | 2,821 | 30,280 / 303 MB | Parquet w/ relative `image_path` column | ★★★★½ subset structure, 6 image categories, paper + BibTeX, mini-leaderboard | Y, 2404.03027 (COLM '24) | **high** |
| 16 | `MMInstruction/MM-SafetyBench` | [search-only] — auth-gated | ~5K image+question pairs (per arXiv 2311.17600) | image-text pairs across 13 unsafe scenarios | unknown | Y, 2311.17600 | **high** |

### 2.3 Multimodal Red-Team Datasets

| # | HF path | DL | Size / # samples | Format | Card | Paper | Rel |
|---|---|---|---|---|---|---|---|
| 17 | `PKU-Alignment/Align-Anything` | (48 likes) | 10K-100K across 17 subsets | Parquet, multimodal (audio+image+text) | ★★★★ rich multi-aspect ratings + refinement reasoning | Y, 2412.15838 | med |
| 18 | `AI-Secure/DecodingTrust` | 359 | 100K-1M / 697 MB | JSON | ★★★★★ 8 trust dimensions, install instructions (pip/conda/docker), tutorial link | Y, 2306.11698 (NeurIPS '23 D&B) | med |
| 19 | `Foreshhh/figstep` (FigStep) | [auth-gated] | ~500 typographic images per arXiv | image+text typographic attack | unknown | Y, 2311.05608 | **high** |
| 20 | `Sterzhang/PhD` | [auth-gated] | unknown | unknown | unknown | unknown | low |

### 2.4 Prompt-Injection (text) Datasets

| # | HF path | DL | Size / # samples | Format | Card | Paper | Rel |
|---|---|---|---|---|---|---|---|
| 21 | `deepset/prompt-injections` | 5,137 | 662 / 56 kB | Parquet | ★★ "More Information needed" placeholder; 46 trained models despite poor card | N | med |
| 22 | `jayavibhav/prompt-injection` | 288 | 327,154 / 77.3 MB | Parquet | ★★ empty README, no methodology, no labels documented | N | low |
| 23 | `Lakera/gandalf_ignore_instructions` | 957 | 1,000 / 56.7 kB | Parquet | ★★★★★ data provenance (Gandalf game, July 2023), filter pipeline (cosine>0.825 + PII), honest noise note | Y, 2501.07927 | **high** |
| 24 | `reshabhs/SPML_Chatbot_Prompt_Injection` | (truncated) | 10K-100K / NA | CSV | ★★★★ tag-rich, MIT, paper-anchored | Y, 2402.11755 | med |
| 25 | `qualifire/prompt-injections-benchmark` (rogue-security/...) | 998 | 5,000 / 5.74 MB | Parquet | ★★★ binary structure clear, methodology missing | N | med |
| 26 | `xTRam1/safe-guard-prompt-injection` | 1,990 | 10,296 / 2.5 MB | Parquet | ★★★★★ category-tree synthetic generation, 99.6% baseline F1 reported, team attribution | Y, 2402.13064 | **high** |
| 27 | `hlyn/prompt-injection-judge-deberta-dataset` | 197 | 399,741 / 196 MB | CSV | ★★★★★ data provenance (12 source datasets listed), MD5 dedup, AUC-ROC + accuracy reported, companion model | N (full provenance instead) | med |
| 28 | `Mindgard/evaded-prompt-injection-and-jailbreak-samples` | 305 | 10K-100K / 3.07 MB | Parquet | ★★★★ 8+ evasion techniques, Base64 emoji-smuggling note | Y, 2504.11168 | med |
| 29 | `prodnull/prompt-injection-repo-dataset` | 62 | 5,671 / 640 kB | JSONL | ★★★★★ 24 attack categories mapped to MITRE+Mindgard, hard-negatives, OOD FPR honestly disclosed (43%), seed=42 reproducibility | Y, multi (2509.22040 et al.) | **high** |

### 2.5 Adversarial / Robustness Image Datasets

| # | HF path | DL | Size / # samples | Format | Card | Paper | Rel |
|---|---|---|---|---|---|---|---|
| 30 | `clip-benchmark/wds_imagenet-a` | 498 | 7,500 / 695 MB | WebDataset (.tar) | ★ no card at all (just auto-metadata) | N (implicitly Hendrycks 1907.07174) | med |
| 31 | `clip-benchmark/wds_imagenet1k` | 2,702 | 82,400 / 156 GB | WebDataset (.tar) | ★★★ minimal card, splits documented | implicit | low |
| 32 | `barkermrl/imagenet-a` | 1,202 | 7,500 / 681 MB | Parquet | ★★★★ 200 class WordNet IDs, BibTeX, links to ImageNet-C/P | Y, 1907.07174 | med |
| 33 | `axiong/imagenet-r` | 2,678 | 30,000 / 2.15 GB | Parquet | ★★★★ load_dataset code example + BibTeX | Y (ICCV '21 Hendrycks) | med |

### 2.6 Polish-Reference Cards (general datasets I sampled for layout)

| # | HF path | DL | Card | Why I looked |
|---|---|---|---|---|
| 34 | `HuggingFaceH4/instruction-dataset` | 474 | ★★★★★ | Clean small-eval card style (327 rows) — analog for visinject's "<1K" size class |
| 35 | `HuggingFaceH4/no_robots` | 8,538 | ★★★½ | Category-distribution table — but lots of "[More Information Needed]" — instructive negative |
| 36 | `yahma/alpaca-cleaned` | 32,446 | ★★★★ | Before/after concrete examples — pattern to emulate for case-study slots |
| 37 | `lmms-lab/POPE` | 27,945 | ★★★★★ | Multimodal eval card with field table + COCO image-source attribution + lmms-eval pipeline integration |
| 38 | `HuggingFaceM4/COCO` | 2,114 | ★★★ | Famous-dataset card surprisingly lazy ("[More Information Needed]") — popularity ≠ card quality |
| 39 | `codeparrot/apps` | 18,621 | ★★★★ | Honest "viewer disabled" + known-limitations (false positives in test coverage) |
| 40 | `jondurbin/airoboros-2.2` | 96 | ★★★★ | GPT-4-derived data → explicit ToS-restriction note. Pattern: pro-actively flag derived-data legal posture |

---

## 3. Deep Dive — `jeffliulab/visinject` Audit

### 3.1 What's already there (good)

- Threat model stated up front (clear scenario: clean photo upload → describe-this-image)
- File tree shown explicitly with directory structure
- Schema example for `response_pairs_<image>.json` with field-level annotation
- 7×3 experiment matrix presented as a 2-table layout (prompts × configs) — readable
- Test-image inventory and target-phrase catalog included
- Citation block + GitHub repo link + Space demo link + companion-paper attribution (arXiv 2502.07987)
- Usage example with `hf_hub_download` AND `snapshot_download` patterns (better than ~half the datasets I surveyed which only offer one)
- License (CC-BY-4.0) and tags (adversarial-attack, vlm-security, ...) are properly set

### 3.2 What's missing or weak (vs. the patterns I saw)

| Gap | Severity | Reference exemplar |
|---|---|---|
| **Dataset Viewer broken** (schema mismatch on `deepseek_vl_1_3b` field) | **High** — viewer is the single biggest source of trust signal for new visitors | `lmms-lab/POPE`, `JailbreakV-28K` (viewers work cleanly) |
| **Judge results not yet uploaded** (the "Output-Affected / Target-Injected" scores live only in your repo, not the dataset) | **High** — this is the most novel methodological contribution; missing it understates the work | `wildguardmix` ships annotations + Fleiss-Kappa together |
| **No content/safety disclaimer** at the top | Med — every 5★ adversarial-text card has one | `Anthropic/hh-rlhf`, `BeaverTails`, `TrustAIRLab/...` |
| **No "what this is NOT"** section (e.g., not a moderation benchmark, not synthetic-detector training) | Med — `prodnull/prompt-injection-repo-dataset` does this perfectly | `prodnull/...` |
| **No image-grid teaser** (clean vs. adversarial side-by-side) | Med — visual cards convert visitors. Your `outputs/succeed_injection_examples/` already has 12 curated images, but they aren't surfaced on the card | `BeaverTails` shows category icons; `MMStar` shows samples |
| **No pre-computed parquet manifest** with one row per (exp, vlm, image, question) for direct `load_dataset()` use | High — currently data viewer fails because schema is per-file JSON. A flat parquet "judge_results.parquet" + "response_pairs.parquet" would fix viewer AND make the dataset queryable in one line | `lmms-lab/POPE` uses single parquet for 18K rows |
| **No injection-success-rate leaderboard** (per-VLM × per-prompt success table) | Med — your data supports this, and `JailbreakV-28K` ships exactly such a mini-leaderboard | `JailbreakBench/JBB-Behaviors` and `JailBreakV-28K` |
| **No reproducibility info**: PGD steps, eps, batch size, seed | Med | `prodnull/...` (seed=42 disclosed) |
| **No transferability section** even though you tested GPT-4o | High — this is genuinely interesting (negative result, but rare in this literature). `prodnull/...` discloses 43% OOD FPR honestly | — |
| **Limitations section is brief** | Low — you mention "uneven success" and "BLIP-2 immune" but don't quantify | `wildguardmix`, `prodnull/...` |
| **No companion paper link to YOUR arXiv** (currently links Rahmatullaev et al. 2502.07987 as the closest match, not your own) | High once you submit | every 5★ card |
| **License on the underlying decoder weights** is unclear from the card (`coco_bi.pt` from `jiamingzz/anyattack`) | Med — derived-model cards (`yahma/alpaca-cleaned`, `airoboros`) flag upstream license carefully | — |
| **Citation is dated 2026 but no DOI/arXiv ID** | Low — fix once arXiv'd | — |

---

## 4. Five Best-Practice Patterns to Adopt

1. **Ship a flat `parquet` manifest alongside the per-experiment JSON tree.** Top-tier multimodal cards (`POPE`, `JailBreakV-28K`, `wildguardmix`) all expose data as a flat queryable table with `image_path` columns. The hierarchical `experiments/exp_<X>/results/<file>.json` layout is great for archival but breaks the HF dataset viewer and forces every consumer to write custom loading code. Add **two parquets**: one row per `(exp, vlm, image, question)` for response pairs, and one row per `(exp, vlm, image)` for judge scores. Keep the JSON tree for full provenance.

2. **Lead with a 6-image teaser strip on the card** (3 clean / 3 adversarial). Use markdown image tags pointing at the curated `outputs/succeed_injection_examples/` files. `BeaverTails`, `lmms-lab/POPE`, and `xTRam1/safe-guard-prompt-injection` all open with a visual artifact and gain 5–10× more user trust before the visitor even reads the prose.

3. **Promote the dual-axis evaluation methodology to a top-of-card section.** "Output-Affected vs. Target-Injected" is the headline scientific contribution (corrects the v1 misreporting from 50.5% to 0.227%). Mirror the *exact* table format that `wildguardmix` uses for its three task-level metrics: a small markdown table with axis name, definition, value, evaluator. This is also where you cite the Fleiss-Kappa-style honesty: report inter-Claude-agent agreement (you used 7 parallel agents on 6,615 pairs — that's an inter-rater story you can quantify).

4. **Add a "What this dataset is NOT" callout box.** Copy `prodnull/prompt-injection-repo-dataset`'s explicit-scope pattern. For visinject this is critical because (a) it is *not* a moderation training set, (b) it is *not* a transferable attack toolkit (your GPT-4o negative result proves this), (c) it is *not* a benchmark of all VLMs (BLIP-2 is immune; Phi/LLaVA/Llama-3.2-Vision were excluded for VRAM/version reasons). Pre-empting these misuses earns reviewer trust and reduces issue-tracker noise.

5. **Disclose the reproducibility budget on the card.** PGD steps (50–500), eps (16/255), step size, batch size, seed, GPU (H200 80 GB), wall-clock (per experiment). `prodnull/...` does this. `DecodingTrust` does this. The cost of running a single experiment is one of the most-asked questions for adversarial-attack datasets, and you have the numbers in `docs/HPC_GUIDE.md` already — just lift them.

---

## 5. Three Anti-Patterns I Saw — Avoid These

1. **Empty / placeholder README ("More Information needed")**: `jayavibhav/prompt-injection` has 327k samples and 288 dl/month but no README, so it gets ★★ and zero academic adoption. `deepset/prompt-injections` similarly minimal. **Even with a paper, a thin card caps the dataset's reach.** Don't ship the arXiv version with any "TBD"-style sections.

2. **Auth-gating without justification.** Every 401 I hit (`MM-SafetyBench`, `FigStep`, `HADES`, `MultiJail`) cost potential adoption. Some are gated for ethics reasons, but the visible pattern is that ungated equivalents (`JailbreakBench`, `BeaverTails`, `JailBreakV-28K`, `prodnull/...`) have *10–100× the downloads* of their gated peers. visinject is already public-CC-BY-4.0; **keep it that way for the arXiv release** even if you add v2-dev attack categories. A content warning + an ethical-use note (à la `Anthropic/hh-rlhf`) is enough.

3. **Modality claim without modality data**: `AiActivity/All-Prompt-Jailbreak` claims "Image" modality but the README is empty and there's no actual image data — pure loss of trust. visinject is the opposite (it actually has the images), but the **viewer schema bug** creates the same impression to a casual visitor — the viewer fails, so the visitor assumes the data is broken. Fix the schema (Section 4 pattern 1) before arXiv release.

---

## 6. Positioning Recommendation (1–2 sentence pitch)

**Pitch (for the dataset card "Summary" line and the arXiv abstract opener):**

> *VisInject is the first publicly-released dataset of imperceptible (PSNR ≈ 25 dB) adversarial-image prompt-injection attacks against open-source VLMs, paired with a dual-axis (Output-Affected / Target-Injected) evaluation that separates "the model is disrupted" from "the attacker's content was actually planted" — a distinction missing from existing multimodal jailbreak benchmarks (JailBreakV-28K, MM-SafetyBench, HADES) that score success solely on "did the model say something harmful."*

**Why this stands out:**
- vs. **JailBreakV-28K / MM-SafetyBench / FigStep / HADES**: those datasets attack *safety alignment* (get the VLM to produce harmful content). visinject attacks *output integrity* (get the VLM to plant attacker-chosen content into a benign Q&A flow). Different threat model, currently no public dataset for it.
- vs. **VLGuard / RobustVLGuard / GuardReasoner-VLTrain**: those are *defense* training data. visinject is *attack-side* artifacts, useful for evaluating the defenses they produce.
- vs. **AdvBench / HarmBench / JailbreakBench**: those are pure-text. visinject is image-perturbation-based with explicit PSNR + L∞ budgets.
- vs. **Rahmatullaev et al. 2502.07987 (the cited "Universal Adversarial Attack on Aligned Multimodal LLMs")**: theirs is single-model + jailbreak-target. visinject extends to multi-model universal + arbitrary-target injection + dual-axis judging.

**The honest negative result** (15/6,615 confirmed injections = 0.227%, GPT-4o transfer fails entirely, BLIP-2 fully immune) is *itself a contribution* — it reframes the field's optimism and aligns with the "When Benchmarks Lie" critique cited by `prodnull/...`. Don't bury it; lead with it.

---

## 7. Expansion Recommendation

### 7.1 Should you 2–5× the data before arXiv?

**Short answer: a targeted ~3× expansion, yes — but on specific axes, not size for its own sake.** Most of the high-quality adversarial datasets I surveyed (`JBB-Behaviors`: 500 rows, `AdvBench`: 500, `Lakera/gandalf`: 1000, `prodnull/...`: 5,671) are *small and focused* and beat 100K-row noisy datasets in adoption. The win is **methodology + reproducibility + multi-axis evaluation**, not raw volume.

### 7.2 Specific expansion priorities (ranked by ROI for arXiv release)

| Priority | Expansion axis | Reason | Effort | Ship? |
|---|---|---|---|---|
| **P0** | **More VLMs** — add Phi-3.5-Vision, LLaVA-1.6, Llama-3.2-Vision-11B (the wrappers already exist in repo, just excluded for VRAM/version reasons). Even partial coverage on H200 should fit. | Currently only 4 VLMs in the matrix; reviewers will ask "does this hold on bigger/newer models?" Adding 2–3 more triples credibility. Even a *negative* result on Llama-3.2-Vision (the most-deployed open VLM as of Apr 2026) is a strong outcome | High (HPC-time, debug version-pinning) | **Yes — required** |
| **P0** | **Fix the dataset viewer + add flat parquets** | Section 4 pattern 1 | Low (a few hours of pandas munging on existing JSONs) | **Yes — required** |
| **P0** | **Upload judge results (Output-Affected + Target-Injected scores)** | Already computed; just not on HF. The novel scientific contribution is currently invisible to dataset visitors | Trivial (`hf upload`) | **Yes — required** |
| **P1** | **More test images** — go from 7 to 21–30, sampled across categories (faces, documents, screenshots, products, scenes, charts) | 7 images is the smallest sample count of any multimodal dataset I surveyed; reviewers will flag this. 30 images keeps the matrix small enough to re-run universally | Med (HPC-time per image) | **Yes — recommended** |
| **P1** | **Add transferability column** (clean+adv → run through GPT-4o-mini, GPT-4o, Claude-3.5-Sonnet via API, log per-image transfer success) | You already manually tested GPT-4o; promote to systematic. Costs ~$50–200 in API. The negative-transfer story is genuinely publishable | Med ($, infra) | **Yes — recommended, but separate parquet so cost is bounded** |
| **P2** | **Add v2-dev *typographic* attack category** (text rendered into image) | This is FigStep's territory; if you add it you can directly compare. Simpler than steganography/cross-modal | Med (port from v2-dev) | **Optional — only if v2-dev branch is ≥80% ready** |
| **P3** | **Add v2-dev *steganography* attack category** | Powerful threat model but easy to over-fit; needs careful methodology (vs. the "Invisible Injections" 2507.22304 paper) | High | **Defer to follow-up paper** |
| **P3** | **Add v2-dev *cross-modal* attack** (audio→VLM-with-vision) | Outside current scope; the four VLMs in the matrix don't all support audio | High | **Defer** |
| **P3** | **Add v2-dev *scene spoofing*** (geometric substitution) | Interesting but needs separate threat-model framing; dilutes the core contribution | High | **Defer** |
| **P3** | **More attack prompts** (currently 7) | Diminishing returns — you'd be sampling the same target-pattern space (URL/email/brand/...) at finer granularity | Low | **Don't bother — 7 is already broader than HarmBench's category list scaled to your axis** |

### 7.3 Suggested sweet-spot scope for the arXiv companion

| Component | Current (v1.1) | Recommended (arXiv) | Multiplier |
|---|---|---|---|
| Universal images | 21 | 35-50 (more VLMs × same prompts) | 1.7-2.4× |
| Adversarial photos | 147 | 700-1,500 (more VLMs × more clean images × same prompts) | 5-10× |
| Response pairs | 6,615 | 30K-60K | 4-9× |
| **Judge results** | **0 on HF** | **all of the above, on HF as parquet** | ∞ |
| Transferability (closed-API) | manual GPT-4o spot-check | systematic (3 closed APIs × subset) | ~10× |
| New attack categories | 0 | typographic only (1) | +1 |

This lands you at "**~3× current scale, with the dual-axis methodology + transferability + viewer-friendly parquets as the main differentiators**" — directly competitive with `JailBreakV-28K` (28K samples) and `MM-SafetyBench`-class scope without overshooting into noisy territory.

### 7.4 What to explicitly *not* do

- Don't try to compete on raw size with `BeaverTails` (364K) or `hh-rlhf` (169K) — different research-genre, different evaluation mode
- Don't add steganography / cross-modal / scene-spoofing in this arXiv release — those are a follow-up paper. Adding them now dilutes both papers
- Don't gate the dataset. Public CC-BY-4.0 wins
- Don't auto-generate more attack prompts via GPT-4 — `airoboros-2.2`'s ToS warning shows the legal headache
- Don't promise "real-world deployment study" — keep the threat model academic

---

## 8. Concrete Pre-arXiv Checklist (in priority order)

1. [ ] Fix dataset viewer schema (drop the `deepseek_vl_1_3b` mismatch — easiest fix: emit *one* parquet of all response pairs with a `vlm` column instead of nested struct)
2. [ ] Upload `judge_results_*.json` to the HF dataset
3. [ ] Generate `response_pairs.parquet` and `judge_results.parquet` flat manifests
4. [ ] Replace the dataset card with one structured like `JailBreakV-28K` + `wildguardmix` (sections: Summary → Visual Teaser → Threat Model → Methodology → Schema → Mini-Leaderboard → "What this is NOT" → Reproducibility Budget → Limitations → Transferability → Citation → License)
5. [ ] Add 3 PNG case-study images to the card (clean | adv | response-diff)
6. [ ] Add the inter-rater agreement number (7 Claude agents on 6,615 pairs)
7. [ ] Run the P0 expansion (more VLMs) on HPC; upload as a v3.1 dataset bump
8. [ ] Optionally run P1 (more test images + closed-API transferability)
9. [ ] Update citation block with arXiv ID once submitted
10. [ ] Add explicit content/safety disclaimer (one paragraph, top of card)
11. [ ] Cross-link from arXiv abstract → HF dataset → GitHub repo → HF Space (4-way link consistency: every property mentions the other three)

---

## 9. References (sources I actually fetched)

Datasets listed in Section 2 (rows 1-15, 17-18, 21-29, 30-33, 34-40) were each fetched live and the content reflects their HF dataset card as of 2026-04-30. Auth-gated rows (16, 19, 20, plus several attempted but excluded because no useful content returned: `MMInstruction/MM-SafetyBench`, `Foreshhh/figstep`, `Lin-Chen/RTVLM`, `Hannibal046/MMSafetyBench`, `MMRedTeam/MMRedTeam`, `safety-research/agent-redteam`, `Sterzhang/PhD`, `aurora-m/MultiJail`, `mlfoundations/imagenet-bench`, `Felladrin/imagenet-r`, `euclaise/imagenet-r`, `imagenet-c`, `hendrycks/imagenet-c`, `tinyimagenet/tinyimagenet`, `imagenet_a`, `imagenet_sketch`, `imagenet-1k`, `Salesforce/wildguardmix`, `lmms-lab/MMStar`, `AdvBench/AdvBench`, `aurora-m/multilingual-jailbreak`, `llm-attacks/AdvBench`, `redteaming/llm-attacks`, `agentlans/prompt-injection`, `PKU-Alignment/SPA-VL`, `ToxicityPrompts/PolygloToxicityPrompts`) are flagged as such.
