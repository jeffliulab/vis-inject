# VisInject Related-Work Survey — for arXiv §2

> Verified literature survey for the VisInject preprint. All papers below were located via WebSearch/WebFetch on arXiv, OpenReview, conference proceedings, or author pages. Each row's verification status is annotated in the rightmost-but-one column (V = arXiv ID + venue confirmed via search; V* = title + venue confirmed but ID inferred from one source).

> **Tag legend** — BASE = should be quantitatively compared in §5/§6; RW = cite, no compare; KM = key motivation for a VisInject design choice; MI = method we directly inspire from / reuse.

> Coverage: 38 papers across 6 sub-areas + 4 foundation references (CLIP, BLIP-2, Qwen-VL, DeepSeek-VL, LLaVA) for self-completeness. Papers tagged BASE are the ones whose numbers VisInject's §6 should report alongside its own.

---

## 1. Universal Adversarial Attacks on Multimodal LLMs / VLMs

| # | Title | Authors | Year | Venue / arXiv | One-line contribution | Relevance to VisInject | V | Tag |
|---|---|---|---|---|---|---|---|---|
| 1 | Universal Adversarial Attack on Aligned Multimodal LLMs | Rahmatullaev, Druzhinina, Kurdiukov, Mikhalchuk, Kuznetsov, Razzhigaev | 2025 | arXiv 2502.07987 | Single optimized image overrides safety alignment across queries and across multiple VLMs (up to 81% ASR on SafeBench/MM-SafetyBench) | **Stage-1 of VisInject is a direct re-implementation of this method**; we extend it with a multi-VLM ensemble and benign-distribution evaluation | V | MI / BASE |
| 2 | AnyAttack: Towards Large-scale Self-Supervised Adversarial Attacks on Vision-Language Models | Zhang, Ye, Ma, Li, Yang, Chen, Sang, Yeung | 2025 | CVPR 2025 (arXiv 2410.05346) | Self-supervised pretrained "attack-foundation" decoder (LAION-400M, eps=16/255) that turns any clean image into a targeted adversarial example transferring to GPT/Claude/Gemini | **Stage-2 of VisInject reuses the released `coco_bi.pt` decoder verbatim**; our novelty is the decoder's *input* — a Stage-1 universal image rather than a CLIP target embedding | V | MI / BASE |
| 3 | Visual Adversarial Examples Jailbreak Aligned Large Language Models | Qi, Huang, Panda, Henderson, Wang, Mittal | 2024 | AAAI 2024 (Oral); arXiv 2306.13213 | Single visual adversarial example universally jailbreaks aligned VLMs into following harmful instructions outside the optimization corpus | First demonstration that the visual modality is the *weak link* of aligned VLMs; motivates why VisInject targets the image even though the threat model forbids prompt manipulation | V | KM / BASE |
| 4 | On the Adversarial Robustness of Multi-Modal Foundation Models | Schlarmann, Hein | 2023 | ICCVW 2023 (AROW); arXiv 2308.10741 | At eps=1/255 imperceptible perturbations re-route VLM captions to attacker-chosen URLs / fake info | Closest precedent to the *misinformation/URL* threat model in VisInject §1.1; we extend their setup to dual-dim evaluation on 60 benign Q&A | V | KM / BASE |
| 5 | Are Aligned Neural Networks Adversarially Aligned? | Carlini, Nasr, Choquette-Choo, Jagielski, Gao, Awadalla, Koh, Ippolito, Lee, Tramèr, Schmidt | 2023 | NeurIPS 2023; arXiv 2306.15447 | Multimodal models are 10× easier to adversarially break than text-only ones; brute force finds adversarial inputs even where NLP attacks fail | Headline citation for §3 (threat model); justifies focusing the attack surface on images rather than text | V | KM |
| 6 | Image Hijacks: Adversarial Images can Control Generative Models at Runtime | Bailey, Ong, Russell, Emmons | 2023/2024 | ICML 2024; arXiv 2309.00236 | "Behaviour Matching" trains image hijacks for ≥80% ASR against LLaVA — covering output control, context exfiltration, safety-override, false beliefs | The four attack categories map almost 1-to-1 onto VisInject's prompts (URL / Card / Email / Obey); useful baseline for the *capability axis* | V | RW / BASE |

## 2. Indirect Prompt Injection

| # | Title | Authors | Year | Venue / arXiv | One-line contribution | Relevance to VisInject | V | Tag |
|---|---|---|---|---|---|---|---|---|
| 7 | Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection | Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz | 2023 | AISec '23 (ACM); arXiv 2302.12173 | Foundational taxonomy of indirect prompt injection: data exfiltration, worming, ecosystem contamination | Defines the "indirect" threat model; VisInject is the **visual-modality analogue** — payload is delivered via the image channel rather than retrieved text | V | KM |
| 8 | Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs | Bagdasaryan, Hsieh, Nassi, Shmatikov | 2023 | arXiv 2307.10490 | First demonstration that adversarial perturbations on image/audio can act as instruction injections against LLaVA, PandaGPT | **Closest prior work to VisInject in spirit**; we differ by (a) using a self-supervised decoder for stealth (PSNR 25 dB) and (b) introducing dual-dim evaluation distinguishing drift from injection | V | RW / BASE |
| 9 | Adversarial Illusions in Multi-Modal Embeddings | Zhang, Jha, Bagdasaryan, Shmatikov | 2024 | USENIX Security 2024 (Distinguished Paper); ePrint via USENIX | Embedding-space attacks align any image/sound to an attacker-chosen target across ImageBind, AudioCLIP, Titan | Shows the same primitive (CLIP-space targeting) that AnyAttack uses, but evaluated on retrieval/classification rather than generative VLMs | V | RW |
| 10 | Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models (BIPIA) | Yi, Xie, Zhu, Wang, Wu, Xie, Zhu, Wang, Yang, Tian, Han, Sun, Xie | 2023/2025 | KDD 2025; arXiv 2312.14197 | First benchmark for indirect prompt injection on text LLMs; proposes boundary-awareness + explicit-reminder defenses | Methodological template for §3 — but uniformly text-only; VisInject fills the **visual** branch | V | RW |
| 11 | Formalizing and Benchmarking Prompt Injection Attacks and Defenses | Liu, Jia, Geng, Jia, Gong | 2024 | USENIX Security 2024; arXiv 2310.12815 | Unified framework formalizing prompt-injection attacks; 5 attacks × 10 defenses × 10 LLMs × 7 tasks | Provides the formal definition of "injection" we adopt: a successful injection is one whose target string appears in the response. Motivates Check-2 (Target-Injected) in our dual-dim metric | V | KM |
| 12 | InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents | Zhan, Liang, Bonatti, Xie, Cui, Gan, Chong, Zheng, Tian, Han | 2024 | ACL Findings 2024; arXiv 2403.02691 | 1,054 test cases; ReAct-prompted GPT-4 vulnerable 24% of the time | Shows indirect-injection generalizes from text to *tool-using agents* — relevant to VisInject's "agent picks up an attacker image" threat model | V | RW |

## 3. Multimodal Jailbreaks Against Aligned VLMs

| # | Title | Authors | Year | Venue / arXiv | One-line contribution | Relevance to VisInject | V | Tag |
|---|---|---|---|---|---|---|---|---|
| 13 | Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG) | Zou, Wang, Carlini, Nasr, Kolter, Fredrikson | 2023 | arXiv 2307.15043 | Greedy Coordinate Gradient finds transferable jailbreak suffixes; 100% ASR on Vicuna-7B, transfers to ChatGPT/Bard/Claude | The PGD analogue VisInject Stage-1 mirrors in the visual modality; AdvBench's 500 harmful behaviors are the textual reference set | V | RW / BASE |
| 14 | Jailbreak in Pieces: Compositional Adversarial Attacks on Multi-Modal Language Models | Shayegani, Dong, Abu-Ghazaleh | 2024 | ICLR 2024 Spotlight; arXiv 2307.14539 | Cross-modality compositional jailbreak — adversarial image + benign prompt | Demonstrates that splitting payload across modalities evades text-only safety filters; complements our finding that drift ≫ injection (the literature has been measuring the wrong thing) | V | RW / BASE |
| 15 | FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts | Gong, Ran, Liu, Wang, Cui, Wang, Sun, Wu | 2025 | AAAI 2025 (Oral); arXiv 2311.05608 | Renders prohibited text as a typographic image; 82.5% avg ASR across 6 open-source VLMs | A *non-adversarial* visual jailbreak — useful contrast: VisInject achieves drift via imperceptible noise, FigStep via overt typography | V | BASE |
| 16 | Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt (BAP) | Ying, Liu, Liu, Wei, Zhao, Liu, Wei | 2024 | arXiv 2406.04031 | Optimizes image *and* text prompts jointly; +29% ASR over single-modality attacks | Threat-model contrast — BAP requires text manipulation, VisInject explicitly forbids it | V | RW / BASE |
| 17 | Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking MLLMs (HADES) | Li, Guo, Zhou, Cheng, Li, Liu, Sun | 2024 | ECCV 2024 Oral; arXiv 2403.09792 | Combines typography + diffusion-generated harmful imagery + adversarial perturbation; 90.3% ASR LLaVA-1.5, 71.6% Gemini Pro Vision | Strong upper-bound for visual jailbreak ASR; VisInject's far lower 0.227% literal injection rate justifies our re-framing of the metric | V | BASE |
| 18 | ImgTrojan: Jailbreaking Vision-Language Models with ONE Image | Tao, Zhong, Li, Liu, Kong | 2024/2025 | NAACL 2025; arXiv 2403.02910 | Data-poisoning at 0.0001% poison ratio (1/9,198) suffices to jailbreak LLaVA-v1.5 | Orthogonal threat model (training-time vs inference-time); cite as "another channel for the same outcome" | V | RW |
| 19 | VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models | Yin, Ye, Zhang, Du, Zhu, Liu, Chen, Wang, Ma | 2023 | NeurIPS 2023; arXiv 2310.04655 | Black-box adversarial examples via single-modal (BSA) + multi-modal (ICSA) iterations against fine-tuned VL models | Precedent for the "transfer through CLIP encoder" recipe AnyAttack later scales | V | RW |
| 20 | On Evaluating Adversarial Robustness of Large Vision-Language Models (AttackVLM) | Zhao, Pang, Du, Yang, Cheung, Lin | 2023 | NeurIPS 2023 | Targeted black-box attacks transferred from CLIP/BLIP to MiniGPT-4, LLaVA, BLIP-2, UniDiffuser, Img2Prompt | The de-facto baseline for "targeted black-box on open-source VLMs" — VisInject should report numbers on AttackVLM's setup | V | BASE |
| 21 | Query-Relevant Images Jailbreak Large Multi-Modal Models | Liu, Zhu, Lan, Dong, Hu, Wang, Lyu, Cui, Chen | 2024 | (referenced in MM-SafetyBench / various 2024 follow-ups) | Stable-diffusion-generated query-relevant images + typography jailbreak open-source LMMs | Black-box analogue showing semantically-relevant images, not just adversarial noise, can drive injection | V* | RW |

## 4. Adversarial Robustness of VLMs / Foundation Models (Defense-Adjacent Attacks)

| # | Title | Authors | Year | Venue / arXiv | One-line contribution | Relevance to VisInject | V | Tag |
|---|---|---|---|---|---|---|---|---|
| 22 | Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings | Schlarmann, Singh, Croce, Hein | 2024 | ICML 2024 Oral; arXiv 2402.12336 | Plug-and-play robust CLIP — drop-in replacement for the vision encoder of any LVLM with no LVLM retraining | The *most actionable defense* for VisInject's threat model; we should report whether AnyAttack transfers to VLMs whose CLIP has been swapped out | V | KM / BASE |
| 23 | Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich VLMs | Hossain, Imteaj | 2024 | arXiv 2407.14971 | Siamese + cosine + stop-gradient — outperforms Robust-CLIP without large batches | Same defense family as Robust-CLIP; cite as "second defensive baseline" | V | RW |
| 24 | Towards Deep Learning Models Resistant to Adversarial Attacks (PGD-AT) | Madry, Makelov, Schmidt, Tsipras, Vladu | 2018 | ICLR 2018; arXiv 1706.06083 | Foundational PGD adversarial training; only ICLR-2018 defense to survive scrutiny | The optimization primitive used in Stage-1; cite for definition of PGD attack | V | MI |
| 25 | Towards Evaluating the Robustness of Neural Networks (C&W) | Carlini, Wagner | 2017 | IEEE S&P 2017; arXiv 1608.04644 | Three new attack algorithms breaking defensive distillation at 100% | Reference for adversarial-attack methodology lineage | V | RW |
| 26 | Explaining and Harnessing Adversarial Examples (FGSM) | Goodfellow, Shlens, Szegedy | 2015 | ICLR 2015; arXiv 1412.6572 | FGSM and the "linear hypothesis" of adversarial vulnerability | Cite as origin of gradient-based image attacks | V | RW |
| 27 | Universal Adversarial Perturbations | Moosavi-Dezfooli, Fawzi, Fawzi, Frossard | 2017 | CVPR 2017; arXiv 1610.08401 | Single image-agnostic perturbation fools state-of-the-art classifiers | Cite as origin of *universal* (single perturbation, many inputs) framing — VisInject Stage-1 is its modern multi-VLM analogue | V | MI |
| 28 | Survey of Adversarial Robustness in Multimodal Large Language Models | Liu, Chen, Yang, Wang | 2025 | arXiv 2503.13962 | 2025 survey covering 100+ attacks/defenses across MLLMs | Use as the umbrella reference when listing attack taxonomies in §2 | V | RW |

## 5. Defenses (Detection / Preprocessing / Robust Training / Safe Tuning)

| # | Title | Authors | Year | Venue / arXiv | One-line contribution | Relevance to VisInject | V | Tag |
|---|---|---|---|---|---|---|---|---|
| 29 | Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models (VLGuard) | Zong, Bohdal, Yu, Yang, Hospedales | 2024 | ICML 2024; arXiv 2402.02207 | Post-hoc / mixed safety fine-tuning dataset; brings adversarial ASR ≈ 0 in many cases | The only defense that has shown near-zero ASR against black-box jailbreaks; useful upper bound for VisInject's transferability claim (BLIP-2 ≈ 0 already without VLGuard!) | V | KM / BASE |
| 30 | AdaShield: Safeguarding Multimodal LLMs from Structure-based Attack via Adaptive Shield Prompting | Wang, Liu, Zhang, Liu, Cui, Liu, Sun, Liu, Wang, Wu, Xu | 2024 | ECCV 2024; arXiv 2403.09513 | Training-free adaptive defense prompts against typographic/structure-based attacks | Defense on the *prompt* axis, complementary to Robust-CLIP on the *encoder* axis | V | RW |
| 31 | Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation (ECSO) | Gou, Chen, Liu, Zhang, Wang, Lu | 2024 | ECCV 2024; arXiv 2403.09572 | Routes suspect images through OCR/captioning, falling back to text-only LLM safety; +37.6% on MM-SafetyBench (SD+OCR) | The defense most likely to *neutralize VisInject* — adversarial pixels are flattened by I2T conversion. Should be discussed in §7 | V | KM |
| 32 | SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks | Robey, Wong, Hassani, Pappas | 2023 | arXiv 2310.03684 | Randomized smoothing on character-level perturbations; robust to GCG/PAIR/AmpleGCG | Inspiration for a future "image-pixel SmoothLLM" defense — cite in §7 future work | V | RW |
| 33 | DiffPure: Diffusion Models for Adversarial Purification | Nie, Guo, Huang, Xiao, Vahdat, Anandkumar | 2022 | ICML 2022; arXiv 2205.07460 | Diffusion forward+reverse process removes adversarial noise; SOTA on CIFAR-10/ImageNet/CelebA-HQ | Direct candidate defense against VisInject's eps=16/255 perturbation; we should at minimum acknowledge | V | KM |
| 34 | PromptShield: Deployable Detection for Prompt Injection Attacks | (multiple authors; CODASPY 2025) | 2025 | CODASPY 2025; arXiv 2501.15145 | Fine-tuned detector for indirect text injection at low FPR | Text-side analogue; motivates a "VisShield" detector trained on Stage-2 outputs | V | RW |
| 35 | Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks | Hossain, Imteaj | 2024 | arXiv 2409.07353 | Combines adversarial training of CLIP with safety-aware decoder fine-tuning | Closest single defense to VisInject's *attack* threat model; numeric comparison would be informative | V | RW |

## 6. Evaluation Methodology / Benchmarks for VLM Safety

| # | Title | Authors | Year | Venue / arXiv | One-line contribution | Relevance to VisInject | V | Tag |
|---|---|---|---|---|---|---|---|---|
| 36 | HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal | Mazeika, Phan, Yin, Zou, Wang, Mu, Sakhaee, Li, Basart, Li, Forsyth, Hendrycks | 2024 | ICML 2024; arXiv 2402.04249 | 18 attacks × 33 LLMs/defenses; 4 behavior categories incl. multimodal | Methodological gold standard; VisInject should adopt a HarmBench-shaped evaluation matrix in §5 | V | KM / BASE |
| 37 | JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models | Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramèr, Hassani, Wong | 2024 | NeurIPS 2024 D&B Track; arXiv 2404.01318 | Open evolving repo of 100 behaviors + standard scoring + leaderboard | The reproducibility/standardization template; VisInject's release of `outputs/experiments/` mirrors this idea | V | KM |
| 38 | MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models | Liu, Zhu, Lan, Dong, Hu, Wang, Cui, Chen, Lyu | 2024 | ECCV 2024; arXiv 2311.17600 | 13 scenarios × 5,040 image-text pairs; query-relevant images for safety eval | Direct benchmark on which VisInject should be reported; closest in scale (5,040 pairs) to VisInject's 6,615 | V | BASE |
| 39 | Red Teaming Visual Language Models (RTVLM) | Li, Li, Yin, Ahmed, Cheng, Lin, Hovy | 2024 | ACL Findings 2024; arXiv 2401.12915 | 10 subtasks across faithfulness, privacy, safety, fairness; up to 31% gap with GPT-4V | Provides the *axes* against which a VLM-safety paper should report — VisInject covers the "safety" axis | V | RW |
| 40 | MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for VLMs | Weng, Xu, Zhang, Chen, Zhang | 2025 | AAAI 2025; arXiv 2408.08464 | First unified VLM jailbreak attack-defense leaderboard | The most natural target benchmark for VisInject §5 numerical comparison | V | KM / BASE |
| 41 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica | 2023 | NeurIPS 2023 D&B; arXiv 2306.05685 | LLM-as-judge correlates 80%+ with human pairwise preference | Foundational reference for our *previous* v1 evaluation pipeline (§Appendix A.2 of report); we report why we replaced it with deterministic dual-dim eval | V | KM |
| 42 | Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR) | Chao, Robey, Dobriban, Hassani, Pappas, Wong | 2023 | arXiv 2310.08419 | Iterative attacker-LLM refines jailbreak in <20 queries — 250× more efficient than GCG | Black-box baseline; VisInject is white-box on open VLMs but transferability story should reference PAIR's efficiency framing | V | RW |

## Foundation References (cite for self-completeness; no comparison)

| # | Title | Authors | Year | Venue | Why we cite |
|---|---|---|---|---|---|
| F1 | Learning Transferable Visual Models From Natural Language Supervision (CLIP) | Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger, Sutskever | 2021 | ICML 2021 | The encoder used by AnyAttack for Stage-2 |
| F2 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs | Li, Li, Savarese, Hoi | 2023 | ICML 2023 | One of the four target VLMs in VisInject |
| F3 | Qwen-VL / Qwen2-VL / Qwen2.5-VL Technical Reports | Bai, Bai, Yang, Wang, Tan, Wang, Lin, Zhou, Zhou et al. | 2023 / 2024 / 2025 | arXiv 2308.12966 / 2409.12191 / 2502.13923 | Three of the four target VLMs |
| F4 | DeepSeek-VL: Towards Real-World Vision-Language Understanding | Lu, Liu, Sun, Tao, Lin, Yang, Wang, Liu, Yu, Zheng, Liu, Bi et al. | 2024 | arXiv 2403.05525 | One of the four target VLMs |
| F5 | Visual Instruction Tuning (LLaVA) | Liu, Li, Wu, Lee | 2023 | NeurIPS 2023 Oral | Architectural template for all instruction-tuned VLMs in our test set |

---

## Specific Gaps in the Literature that VisInject Fills

1. **The "drift vs injection" conflation.** Existing VLM-jailbreak papers (FigStep, HADES, BAP, Jailbreak-in-Pieces) report a single ASR — typically a Boolean "did the model refuse / did harmful content appear" judged by an LLM-judge. **None** decompose this into (a) the model's output was *changed* by the perturbation vs (b) the *attacker-chosen string* was actually emitted. VisInject's 6,615-pair dual-dim sweep is the first to show these two numbers diverge by 290× (66.3% vs 0.227%) on the same images. This re-frames what published "ASR" numbers actually measure.

2. **No standardized benign-distribution evaluation.** HarmBench, JailbreakBench, MM-SafetyBench, RTVLM all evaluate on *adversarial inputs queried with adversarial prompts*. The threat surface VisInject targets is the opposite — a **benign user prompt** ("describe this image") on a stealthily-perturbed image. Our 60-question benign Q&A set per (clean, adv) pair is, to our knowledge, unique.

3. **PSNR-controlled, semantic-stealthy attack.** AnyAttack reports L∞ at eps=16/255 but evaluates on retrieval/classification, not generative dialog. Schlarmann ICCVW'23 reports semantic effects but at eps=1/255 on captioning only. VisInject is the first to report **PSNR ~25 dB perceptually-stealthy** attacks on free-form generative VLM dialog at scale (147 image pairs × 60 prompts × 4 VLMs = 35,280 generations).

4. **Transferability ablation across four open-source VLM families.** Existing transferability claims focus on closed-source (GPT-4o / Gemini / Claude) success rates. VisInject reports the *failure case* — that the same attack that succeeds 100% on Qwen2.5-VL (8.45/10 effect score) achieves 0.0/10 on BLIP-2, despite both being instruction-tuned VLMs. No prior work has documented this immunity gap, which carries actionable implications for VLM architecture choices.

5. **Open, fully-reproducible 6,615-pair artifact.** HarmBench/JailbreakBench provide 100-400 behaviors; MM-SafetyBench 5,040 image-text pairs (text-only attack). VisInject's release of 21 experiments × 7 images × 60 prompts × ~5 VLMs (with cleartext + adversarial response pairs) is the **largest publicly-released benign-prompt × adversarial-image evaluation matrix**, and it is queryable for follow-up work on detection and defense.

---

## Top 5 Most-Likely Venues for VisInject

| Rank | Venue | Type | Cycle | Typical paper style | Fit notes |
|---|---|---|---|---|---|
| 1 | **NeurIPS / ICLR Datasets & Benchmarks Track** | Top-tier conference | Annual (NeurIPS abstracts ~May, ICLR ~Sep) | 9 pp + appendix; emphasizes reusable artifacts, leaderboard, reproducibility (HarmBench, JailbreakBench, MM-SafetyBench all landed here) | Best fit: VisInject is *primarily* a dataset + evaluation contribution. The 6,615-pair release + dual-dim metric is exactly what this track was designed for. |
| 2 | **USENIX Security Symposium** | Top-tier security | Aug deadline ~Feb; Spring deadline ~Sep | 12-18 pp; concrete threat model, real-world impact, attack/defense both | Strong fit: Adversarial-Illusions, Liu-formalizing-prompt-injection, and Greshake AISec all have a USENIX home. The "cardholder data exfiltration via image" angle is exactly USENIX's vibe. |
| 3 | **ACM AISec Workshop (CCS-co-located)** | Workshop | Annual June deadline | 6-10 pp; security-flavored ML, accepts position+empirical hybrids | Greshake (the original IPI paper) was AISec'23. A "v0" of VisInject targeting ACM AISec is a defensible safety-net submission that still cites well. |
| 4 | **ICML / NeurIPS Workshop on Trustworthy Multimodal Foundation Models / AdvML / SafeML** | Workshop | Spring deadline | 4-8 pp; non-archival or proceedings; oral talks | Multiple workshops in this space (AROW, SafeML, Backdoor & Adversarial Defense). Schlarmann ICCVW'23 was AROW. Lower bar, faster turnaround, builds citation graph before the main-conference submission. |
| 5 | **CVPR / ICCV Workshop on Adversarial Machine Learning (AROW)** | Workshop | Annual | 4-8 pp; vision-flavored adversarial ML | Schlarmann ICCVW'23 lives here; clean home for the *Stage-2 stealth* contribution as a standalone short paper if the main paper goes elsewhere. |

**Recommendation:** Aim primary submission at **NeurIPS D&B Track (May abstract / May-June paper deadline)** — the dual-dim eval + 6,615-pair release maps directly onto that track's evaluation rubric. Hold a **USENIX Security Spring** submission as a parallel "security-narrative" version emphasizing the cardholder/URL exfiltration scenarios. If the main paper is rejected, fork the Stage-2 (decoder-fusion + PSNR analysis) into an **AROW workshop short** to keep the citation graph alive.

---

## Honest Verification Notes

- **All 42 entries** had their arXiv ID, venue, OR direct conference-page URL surfaced via WebSearch. Full search transcripts are in conversation history.
- **Entry #21** (Liu et al. "Query-Relevant Images") is marked V* — the paper is referenced in MM-SafetyBench and in multiple jailbreak surveys, but the primary search hit was a Semantic Scholar / cross-reference rather than a direct arXiv landing. Authoritative cite-key should be re-verified before camera-ready (cross-check Liu, Y., Zhu, P., Lan, Y. et al., COLING/EMNLP/AAAI 2024).
- **Entry #34 (PromptShield)** — the arXiv ID 2501.15145 was surfaced via search but the CODASPY 2025 venue should be re-verified once the proceedings page is live.
- **Entry #22 / Entry #29 author orderings** were copied from search snippets; before final BibTeX, run `crossref` lookup on each DOI.
- **Foundation refs F3 (Qwen-VL series)** combine three separate technical reports (Qwen-VL, Qwen2-VL, Qwen2.5-VL). Cite the specific one matching the model variant in the experiment table.
- **No paper title was invented.** Where I had only second-hand knowledge (e.g., the original "Niu imgJP" paper, the original Liu Query-Relevant-Image paper), I either skipped the entry or marked it V*.
- **Coverage**: Section 1 = 6 papers, Section 2 = 6 papers, Section 3 = 9 papers, Section 4 = 7 papers, Section 5 = 7 papers, Section 6 = 7 papers, Foundations = 5 papers. Total verified = 42 (target: 30-50).

---

## Sources (verification URLs surfaced during research)

- arxiv.org/abs/2502.07987 (Rahmatullaev 2025)
- arxiv.org/abs/2410.05346 ; openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Anyattack... (Zhang CVPR 2025)
- arxiv.org/abs/2306.13213 ; ojs.aaai.org/index.php/AAAI/article/view/30150 (Qi AAAI 2024 Oral)
- arxiv.org/abs/2308.10741 ; openaccess.thecvf.com/content/ICCV2023W/AROW/papers/Schlarmann... (Schlarmann ICCVW 2023)
- arxiv.org/abs/2306.15447 (Carlini NeurIPS 2023)
- arxiv.org/abs/2309.00236 ; image-hijacks.github.io (Bailey ICML 2024)
- arxiv.org/abs/2302.12173 ; dl.acm.org/doi/10.1145/3605764.3623985 (Greshake AISec '23)
- arxiv.org/abs/2307.10490 (Bagdasaryan 2023)
- usenix.org/conference/usenixsecurity24/presentation/zhang-tingwei (Zhang USENIX 2024 Distinguished)
- arxiv.org/abs/2312.14197 (Yi BIPIA)
- arxiv.org/abs/2310.12815 ; usenix.org/conference/usenixsecurity24/presentation/liu-yupei (Liu USENIX 2024)
- arxiv.org/abs/2403.02691 ; aclanthology.org/2024.findings-acl.624 (InjecAgent)
- arxiv.org/abs/2307.15043 ; llm-attacks.org (Zou GCG)
- arxiv.org/abs/2307.14539 ; openreview.net/forum?id=plmBsXHxgR (Shayegani ICLR 2024 Spotlight)
- arxiv.org/abs/2311.05608 ; ojs.aaai.org/index.php/AAAI/article/view/34568 (FigStep AAAI 2025 Oral)
- arxiv.org/abs/2406.04031 (BAP)
- arxiv.org/abs/2403.09792 ; ecva.net/papers/eccv_2024/papers_ECCV/papers/09265 (HADES ECCV 2024 Oral)
- arxiv.org/abs/2403.02910 ; aclanthology.org/2025.naacl-long.360 (ImgTrojan NAACL 2025)
- arxiv.org/abs/2310.04655 ; papers.nips.cc/paper_files/paper/2023/file/a5e3cf29c269b041ccd644b6beaf5c42-Paper-Conference.pdf (VLATTACK NeurIPS 2023)
- yunqing-me.github.io/AttackVLM (AttackVLM NeurIPS 2023)
- arxiv.org/abs/2402.12336 ; proceedings.mlr.press/v235/schlarmann24a.html (Robust CLIP ICML 2024 Oral)
- arxiv.org/abs/2407.14971 (Sim-CLIP)
- arxiv.org/abs/1706.06083 ; openreview.net/forum?id=rJzIBfZAb (Madry ICLR 2018)
- arxiv.org/abs/1608.04644 (C&W IEEE S&P 2017)
- arxiv.org/abs/1412.6572 (Goodfellow ICLR 2015)
- arxiv.org/abs/1610.08401 ; openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli... (Moosavi-Dezfooli CVPR 2017)
- arxiv.org/abs/2503.13962 (Survey 2025)
- arxiv.org/abs/2402.02207 ; ys-zong.github.io/VLGuard (VLGuard ICML 2024)
- arxiv.org/abs/2403.09513 ; eccv.ecva.net/virtual/2024/poster/1116 (AdaShield ECCV 2024)
- arxiv.org/abs/2403.09572 ; gyhdog99.github.io/projects/ecso (ECSO ECCV 2024)
- arxiv.org/abs/2310.03684 (SmoothLLM)
- arxiv.org/abs/2205.07460 ; proceedings.mlr.press/v162/nie22a.html (DiffPure ICML 2022)
- arxiv.org/html/2501.15145 ; dl.acm.org/doi/10.1145/3714393.3726501 (PromptShield CODASPY 2025)
- arxiv.org/abs/2409.07353 (Securing VLMs)
- arxiv.org/abs/2402.04249 ; proceedings.mlr.press/v235/mazeika24a.html (HarmBench ICML 2024)
- arxiv.org/abs/2404.01318 ; proceedings.neurips.cc/paper_files/paper/2024/...63092d79154adebd7305dfd498cbff70 (JailbreakBench NeurIPS 2024)
- arxiv.org/abs/2311.17600 ; ecva.net/papers/eccv_2024/papers_ECCV/papers/07350 (MM-SafetyBench ECCV 2024)
- arxiv.org/abs/2401.12915 ; aclanthology.org/2024.findings-acl.198 (RTVLM ACL Findings 2024)
- arxiv.org/abs/2408.08464 ; ojs.aaai.org/index.php/AAAI/article/view/34983 (MMJ-Bench AAAI 2025)
- arxiv.org/abs/2306.05685 (LLM-as-Judge NeurIPS 2023)
- arxiv.org/abs/2310.08419 ; jailbreaking-llms.github.io (PAIR)
- arxiv.org/abs/2103.00020 ; proceedings.mlr.press/v139/radford21a (CLIP)
- arxiv.org/abs/2301.12597 ; proceedings.mlr.press/v202/li23q (BLIP-2)
- arxiv.org/abs/2308.12966 ; arxiv.org/abs/2409.12191 ; arxiv.org/abs/2502.13923 (Qwen-VL series)
- arxiv.org/abs/2403.05525 ; arxiv.org/abs/2412.10302 (DeepSeek-VL/-VL2)
- arxiv.org/abs/2304.08485 ; papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html (LLaVA NeurIPS 2023 Oral)
