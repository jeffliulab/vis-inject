"""
Quick AnyAttack demo: generate an adversarial image from a single pair.

Usage:
    python demo.py --decoder-path checkpoints/finetuned.pt \
                   --clean-image examples/dog.jpg \
                   --target-image examples/cat.jpg \
                   --output adversarial.png
"""

import argparse
import os
import sys

import torch
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from config import ATTACK_CONFIG
from models import CLIPEncoder, Decoder
from dataset import load_image


def main():
    parser = argparse.ArgumentParser(description="AnyAttack demo")
    parser.add_argument("--decoder-path", type=str, required=True)
    parser.add_argument("--clean-image", type=str, required=True)
    parser.add_argument("--target-image", type=str, required=True)
    parser.add_argument("--output", type=str, default="adversarial.png")
    parser.add_argument("--eps", type=float, default=ATTACK_CONFIG["eps"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading CLIP {ATTACK_CONFIG['clip_model']}...")
    clip_encoder = CLIPEncoder(ATTACK_CONFIG["clip_model"]).to(device)

    print(f"Loading Decoder: {os.path.basename(args.decoder_path)}...")
    decoder = Decoder(embed_dim=ATTACK_CONFIG["embed_dim"]).to(device).eval()
    ckpt = torch.load(args.decoder_path, map_location="cpu")
    state = ckpt.get("decoder_state_dict", ckpt)
    cleaned = {k.removeprefix("module."): v for k, v in state.items()}
    decoder.load_state_dict(cleaned)

    clean = load_image(args.clean_image).to(device)
    target = load_image(args.target_image).to(device)

    with torch.no_grad():
        emb = clip_encoder.encode_img(target)
        noise = decoder(emb)
        noise = torch.clamp(noise, -args.eps, args.eps)
        adv = torch.clamp(clean + noise, 0, 1)

    torchvision.utils.save_image(adv[0], args.output)

    psnr = -10 * torch.log10(torch.mean((clean - adv) ** 2)).item()
    print(f"Saved: {args.output}")
    print(f"Noise L-inf: {noise.abs().max().item():.4f} (budget: {args.eps:.4f})")
    print(f"PSNR: {psnr:.1f} dB")


if __name__ == "__main__":
    main()
