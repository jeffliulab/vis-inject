"""
VisInject Web Demo — Unified inference UI for three VLM models.
Pick a model and an image from demo_images/, then click Run Inference.

Usage:
    python demo_web.py
    # Open http://127.0.0.1:7860 in your browser
"""

import os
import sys
import importlib
import gc
import logging
import time
import traceback

import gradio as gr
import torch
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_IMAGES_DIR = os.path.join(SCRIPT_DIR, "demo_images")

logger = logging.getLogger("demo_web")

# ── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "BLIP-2 (blip2-opt-2.7b)": {
        "dir": os.path.join(SCRIPT_DIR, "demo1_BLIP2"),
        "default_question": "What do you see in this image?",
    },
    "DeepSeek-VL (1.3B)": {
        "dir": os.path.join(SCRIPT_DIR, "demo2_DeepSeekVL_1"),
        "default_question": "Describe this image.",
    },
    "Qwen2.5-VL (3B)": {
        "dir": os.path.join(SCRIPT_DIR, "demo3_Qwen_2_5_VL_3B"),
        "default_question": "Describe this image.",
    },
}

_active_model_key: str | None = None
_active_model = None

# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_image_choices() -> list[str]:
    if not os.path.isdir(DEMO_IMAGES_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(DEMO_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )


def _unload_active_model():
    global _active_model_key, _active_model
    if _active_model is not None:
        logger.info("Unloading model: %s", _active_model_key)
        del _active_model
        _active_model = None
        _active_model_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_via_module(model_key: str, device: str):
    """Load a model by importing its demo directory's model_loader.

    For DeepSeek-VL we apply a surgical monkey-patch to torch.linspace
    before the import, forcing it to create CPU tensors.  This bypasses the
    meta-tensor crash that occurs in siglip_vit.py line 391:

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

    When transformers/accelerate have previously set the default device to
    "meta" (e.g. after loading BLIP-2), torch.linspace produces meta tensors
    whose .item() is illegal.  Passing device='cpu' explicitly overrides the
    meta default, so .item() works normally.
    """
    demo_dir = MODEL_REGISTRY[model_key]["dir"]

    _CONFLICT = ["config", "model_loader", "utils", "pgd_attack"]
    saved_modules = {n: sys.modules.pop(n) for n in _CONFLICT if n in sys.modules}
    saved_path = sys.path[:]
    saved_cwd = os.getcwd()

    # DeepSeek-VL compatibility patches for newer transformers:
    #  1. torch.linspace → force CPU to avoid meta-tensor .item() crash
    #  2. Add missing 'all_tied_weights_keys' attr that new transformers expects
    _orig_linspace = torch.linspace
    if "DeepSeek" in model_key:
        def _linspace_cpu(*args, **kwargs):
            kwargs["device"] = "cpu"
            return _orig_linspace(*args, **kwargs)
        torch.linspace = _linspace_cpu

        from deepseek_vl.models.modeling_vlm import MultiModalityCausalLM
        if not hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
            MultiModalityCausalLM.all_tied_weights_keys = {}
        elif isinstance(MultiModalityCausalLM.all_tied_weights_keys, set):
            MultiModalityCausalLM.all_tied_weights_keys = {}

    try:
        sys.path.insert(0, demo_dir)
        os.chdir(demo_dir)
        ml = importlib.import_module("model_loader")
        return ml.load_model(device=device)
    finally:
        torch.linspace = _orig_linspace
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        for n in _CONFLICT:
            sys.modules.pop(n, None)
        sys.modules.update(saved_modules)


def _load_model(model_key: str):
    global _active_model_key, _active_model

    if _active_model_key == model_key and _active_model is not None:
        return _active_model

    _unload_active_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_via_module(model_key, device)

    _active_model_key = model_key
    _active_model = model
    return model


# ── Model-specific inference ──────────────────────────────────────────────────


def _blip2_generate(model, image_path: str, question: str) -> str:
    """BLIP-2 inference with dtype fix.

    Q-Former outputs float32 but language_projection weights are float16.
    We cast explicitly before the projection layer to avoid the dtype mismatch.
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = model._preprocess_pil_to_tensor(image)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            return_dict=True,
        )

        lp_dtype = model.language_projection.weight.dtype
        language_model_inputs = model.language_projection(
            query_outputs.last_hidden_state.to(lp_dtype)
        )

        attention_mask = torch.ones(
            language_model_inputs.shape[:2], dtype=torch.long, device=model.device
        )

        gc_was_on = getattr(model.language_model, "gradient_checkpointing", False)
        if gc_was_on:
            model.language_model.gradient_checkpointing_disable()

        out = model.language_model.generate(
            inputs_embeds=language_model_inputs,
            attention_mask=attention_mask,
            max_new_tokens=80,
            do_sample=False,
        )

        if gc_was_on:
            model.language_model.gradient_checkpointing_enable()

    return model.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def _deepseek_generate(model, image_path: str, question: str) -> str:
    """DeepSeek-VL inference: try native generate() first, fallback to generate_from_tensor.

    Native model.generate(image_path) uses VLChatProcessor + load_pil_images.
    It requires an absolute path because Gradio callbacks may run with a different cwd.
    If native returns empty (silently catches exceptions), we fall back to
    generate_from_tensor (manual path, same as demo2/test_inference.py --mode manual).
    """
    abs_path = os.path.abspath(image_path)

    # 1. Try native path (same as demo2 test_inference --mode native)
    result = model.generate(abs_path, question)
    if result and result.strip():
        return result

    # 2. Fallback: manual tensor path (demo2/test_inference.py --mode manual)
    IMAGE_SIZE = 384
    img = Image.open(abs_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    img_np = np.array(img).astype(np.float32) / 255.0
    image_tensor = (
        torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(model.device)
    )
    return model.generate_from_tensor(image_tensor, question)


def _run_inference(model, model_key: str, image_path: str, question: str) -> str:
    if "BLIP-2" in model_key:
        return _blip2_generate(model, image_path, question)
    if "DeepSeek" in model_key:
        return _deepseek_generate(model, image_path, question)
    # Qwen2.5-VL: model.generate() works correctly as-is
    return model.generate(image_path, question)


# ── Status helpers ────────────────────────────────────────────────────────────

STATUS_LOADING = "Loading model, please wait (first load may take a while) ..."
STATUS_RUNNING = "Running inference ..."
STATUS_DONE = "Done."


def _fmt_error(stage: str, exc: Exception) -> tuple[str, str]:
    detail = "".join(traceback.format_exception(exc))
    return (
        f"Error during {stage}. See output for details.",
        f"[{stage.upper()} FAILED]\n{exc}\n\n{detail}",
    )


# ── Gradio callbacks ──────────────────────────────────────────────────────────


def on_model_change(model_key):
    if model_key and model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]["default_question"]
    return ""


def on_image_change(image_name):
    if not image_name:
        return None
    path = os.path.join(DEMO_IMAGES_DIR, image_name)
    return Image.open(path).convert("RGB") if os.path.isfile(path) else None


def run_inference(model_key, image_name, question):
    """Load model (if needed) and run inference. Yields (status, output)."""
    if not model_key:
        yield "Error: please select a model first.", ""
        return
    if not image_name:
        yield "Error: please select an image first.", ""
        return

    question = question.strip() or MODEL_REGISTRY[model_key]["default_question"]
    image_path = os.path.join(DEMO_IMAGES_DIR, image_name)

    if not os.path.isfile(image_path):
        yield f"Error: image not found — {image_name}", ""
        return

    # Load model
    need_load = _active_model_key != model_key or _active_model is None
    if need_load:
        yield STATUS_LOADING, ""

    t0 = time.time()
    try:
        model = _load_model(model_key)
    except Exception as e:
        yield *_fmt_error("model loading", e),
        return
    load_time = time.time() - t0

    # Run inference
    yield STATUS_RUNNING, ""
    t1 = time.time()
    try:
        result = _run_inference(model, model_key, image_path, question)
    except Exception as e:
        yield *_fmt_error("inference", e),
        return
    infer_time = time.time() - t1

    lines = [result if result else "(empty output)", "", "--- Timing ---"]
    if need_load:
        lines.append(f"Model load : {load_time:.1f}s")
    lines.append(f"Inference  : {infer_time:.1f}s")
    yield STATUS_DONE, "\n".join(lines)


# ── Build UI ──────────────────────────────────────────────────────────────────

image_choices = _get_image_choices()
model_choices = list(MODEL_REGISTRY.keys())

with gr.Blocks(title="VisInject Demo") as app:
    gr.Markdown("# VisInject Web Demo")
    gr.Markdown(
        "Select a vision-language model and an image from `demo_images/`, "
        "then click **Run Inference** to see the model output."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=model_choices, label="Model", value=None, interactive=True
            )
            image_dd = gr.Dropdown(
                choices=image_choices,
                label="Image (demo_images/)",
                value=None,
                interactive=True,
            )
            question_tb = gr.Textbox(
                label="Question",
                placeholder="Leave empty to use the model's default question",
                lines=2,
            )
            run_btn = gr.Button("Run Inference", variant="primary", size="lg")

        with gr.Column(scale=1):
            preview = gr.Image(
                label="Image Preview", type="pil", interactive=False, height=360
            )
            status_tb = gr.Textbox(label="Status", lines=1, interactive=False)
            output_tb = gr.Textbox(label="Model Output", lines=8, interactive=False)

    model_dd.change(on_model_change, model_dd, question_tb)
    image_dd.change(on_image_change, image_dd, preview)
    run_btn.click(
        run_inference,
        inputs=[model_dd, image_dd, question_tb],
        outputs=[status_tb, output_tb],
    )

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
