# `demo/` — Gradio demos

Two separate Gradio apps are provided, differing in what part of the
VisInject pipeline they execute and what hardware they require.

```
demo/
├── README.md          # (this file)
├── space_demo/        # Stripped, CPU-only, HF Space compatible
│   ├── app.py         # Gradio app — Stage 2 fusion only
│   ├── requirements.txt
│   └── README.md
└── full_demo/         # Original full-pipeline demo (local, GPU required)
    ├── web_demo.py    # Gradio app — Stage 1 + Stage 2 + evaluation
    └── README.md
```

## Which one do I want?

| Feature                        | `space_demo`             | `full_demo`           |
| ------------------------------ | ------------------------ | --------------------- |
| Stage 1 (PGD training)         | No (uses precomputed)    | Yes                   |
| Stage 2 (AnyAttack fusion)     | Yes                      | Yes                   |
| VLM-based ASR evaluation       | No                       | No (always offline)   |
| Hardware                       | CPU, 2 vCPU / 16 GB      | GPU, 11+ GB VRAM      |
| HF Space compatible            | Yes                      | No                    |
| Typical latency per generation | ~2–5 s                   | minutes (full mode)   |

- Pick [`space_demo/`](space_demo/README.md) if you just want to try
  the VisInject attack on an image you upload, or if you plan to host
  the demo on Hugging Face Spaces. It reuses the 7 precomputed
  universal adversarial images from the 21-experiment matrix
  (`outputs/experiments/exp_<tag>_2m/universal/`).
- Pick [`full_demo/`](full_demo/README.md) if you want to re-train
  universal images from scratch against arbitrary target phrases and
  arbitrary VLMs. Requires a local GPU.

## Relation to the batch experiments

The production experiments are still driven by
[`scripts/run_experiments.sh`](../scripts/run_experiments.sh) on the
HPC. Neither demo writes to `outputs/experiments/`; both are meant for
exploratory / qualitative use only.
