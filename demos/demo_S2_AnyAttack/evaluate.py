"""
Evaluate AnyAttack adversarial images against target VLMs.

Tests whether adversarial images successfully shift VLM outputs toward
the target image's content (captioning, retrieval similarity).

Usage:
    python evaluate.py --adv-dir outputs/adversarial --target-dir data/target
    python evaluate.py --adv-image adversarial.png --target-image target.jpg
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import EVAL_CONFIG, ATTACK_CONFIG
from models import CLIPEncoder
from dataset import load_image, EVAL_TRANSFORM
from model_registry import get_model_info, get_hf_id


def compute_clip_similarity(clip_encoder, image1: torch.Tensor,
                            image2: torch.Tensor) -> float:
    """Compute cosine similarity between two images in CLIP space."""
    with torch.no_grad():
        e1 = clip_encoder.encode_img(image1)
        e2 = clip_encoder.encode_img(image2)
        e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
        return (e1 * e2).sum(dim=1).mean().item()


def evaluate_captioning(model, processor, adv_image_path: str,
                        target_caption: str, device) -> dict:
    """Generate caption for adversarial image and compare to target."""
    image = Image.open(adv_image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return {
        "adv_image": adv_image_path,
        "generated_caption": caption,
        "target_caption": target_caption,
    }


def load_vlm(model_key: str, device):
    """Load a VLM from model_registry for evaluation."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    info = get_model_info(model_key)
    hf_id = info["hf_id"]
    print(f"Loading VLM: {info['short_name']} ({hf_id})...")

    dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        hf_id, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    return model, processor


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_encoder = CLIPEncoder(ATTACK_CONFIG["clip_model"]).to(device)

    results = {}

    # CLIP similarity evaluation
    if args.adv_image and args.target_image:
        adv = load_image(args.adv_image).to(device)
        target = load_image(args.target_image).to(device)
        clean = load_image(args.clean_image).to(device) if args.clean_image else None

        sim_adv_target = compute_clip_similarity(clip_encoder, adv, target)
        print(f"CLIP Similarity (adv <-> target): {sim_adv_target:.4f}")

        if clean is not None:
            sim_clean_target = compute_clip_similarity(clip_encoder, clean, target)
            sim_clean_adv = compute_clip_similarity(clip_encoder, clean, adv)
            print(f"CLIP Similarity (clean <-> target): {sim_clean_target:.4f}")
            print(f"CLIP Similarity (clean <-> adv):    {sim_clean_adv:.4f}")
            print(f"Similarity shift: {sim_adv_target - sim_clean_target:+.4f}")

        results["clip_similarity"] = {
            "adv_target": sim_adv_target,
        }

    # VLM captioning evaluation
    for vlm_key in args.target_vlms:
        try:
            model, processor = load_vlm(vlm_key, device)
        except Exception as e:
            print(f"[WARN] Could not load {vlm_key}: {e}")
            continue

        if args.adv_image:
            result = evaluate_captioning(
                model, processor, args.adv_image,
                target_caption="(target image content)", device
            )
            print(f"\n[{vlm_key}] Caption: {result['generated_caption']}")
            results[vlm_key] = result

        del model
        torch.cuda.empty_cache()

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AnyAttack")
    parser.add_argument("--adv-image", type=str, default=None)
    parser.add_argument("--target-image", type=str, default=None)
    parser.add_argument("--clean-image", type=str, default=None)
    parser.add_argument("--adv-dir", type=str, default=None)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--target-vlms", nargs="+",
                        default=EVAL_CONFIG["target_vlms"])
    parser.add_argument("--output", type=str, default="outputs/results/eval.json")
    main(parser.parse_args())
