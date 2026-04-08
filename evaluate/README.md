# `evaluate/` — Stage 3 Evaluation Module

VisInject 三阶段流水线的最后一阶：判定攻击是否成功。

## 设计

评估分两步进行，**故意解耦**，因为它们的硬件需求和成本完全不同：

| 子阶段 | 文件 | 运行位置 | 资源 | 输入 | 输出 |
|---|---|---|---|---|---|
| **3a — Pairs 生成** | `pairs.py` | HPC GPU | 1× H200，加载 N 个 VLM | clean 图 + adv 图 + 60 个良性问题 | `response_pairs_<image>.json` |
| **3b — LLM Judge** | `judge.py` | 本地 / 任意 CPU | 仅需 OpenAI / Anthropic API key | response_pairs JSON | `judge_results_<image>.json` |

3a 必须在 HPC 上跑（VLM 需要 GPU），但 3b 不需要 GPU 也不需要 VLM —— 只是把 JSON 喂给 LLM API 打分。这样：
- 3a 跑一次产出的 JSON 可以反复跑 3b（换不同 judge / 不同 prompt 模板）
- 3b 不占 HPC 时间，便于本地迭代评估方法

## CLI 用法

**`pairs.py`**（HPC 端，由 `pipeline.py` 自动调用，也可单独跑）：
```bash
python -m evaluate.pairs \
    --adv-images outputs/experiments/exp_url_2m/adversarial/adv_ORIGIN_dog.png \
    --clean-images images/ORIGIN_dog.png \
    --universal-image outputs/experiments/exp_url_2m/universal/universal_<hash>.png \
    --target-phrase "visit www.example.com"
```

**`judge.py`**（本地端，需要 `.env` 中的 API key）：
```bash
python -m evaluate.judge \
    --pairs-file outputs/experiments/exp_url_2m/results/response_pairs_ORIGIN_dog.json \
    --judges gpt-4o-mini
```

批量跑所有实验：`bash scripts/judge_all.sh --judges gpt-4o-mini`。

## 程序化调用（已被 `pipeline.py` 使用）

```python
from evaluate import generate_response_pairs

result = generate_response_pairs(
    adv_image_path="...",
    clean_image_path="...",
    target_phrase="visit www.example.com",
    target_vlms=["qwen2_5_vl_3b", "blip2_opt_2_7b"],
    output_path="outputs/experiments/exp_url_2m/results/response_pairs_ORIGIN_dog.json",
)
```

`evaluate/__init__.py` 重导出全部公开 API（`generate_response_pairs`, `run_evaluation`, `evaluate_asr`, `evaluate_image_quality`, `evaluate_clip`, `evaluate_captions`），所以历史的 `from evaluate import xxx` 写法继续有效。

## 输出格式

- `response_pairs_<image>.json` —— 见 [`docs/RESULTS_SCHEMA.md`](../docs/RESULTS_SCHEMA.md)
- `judge_results_<image>.json` —— 同上

## 设计注意

- **3a 与训练完全解耦**：`pairs.py` 不知道 universal image 是怎么来的，它只接收 (clean, adv) 路径
- **3b 与 VLM 完全解耦**：`judge.py` 只读 JSON，不依赖 PyTorch 或任何 VLM
- **API key 通过环境变量传**：`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`，从根目录 `.env` 读取
- **评估问题来自 `attack/dataset.py`**（共用同一份 60 个良性问题集，确保训练/评估问题分布一致）
