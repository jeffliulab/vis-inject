"""
Generate adversarial images using a trained AnyAttack Decoder.

Given clean images and target images, produces adversarial images where
the CLIP embedding of the adversarial image matches the target image.

Usage:
    python generate_adv.py --decoder-path checkpoints/finetuned.pt \
                           --clean-dir data/clean --target-dir data/target
"""

import argparse
import os
import sys

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import ATTACK_CONFIG, GENERATE_CONFIG
from models import CLIPEncoder, Decoder
from dataset import make_imagefolder_dataloader, load_image


def load_decoder(decoder_path: str, device: torch.device) -> Decoder:
    """Load trained Decoder from checkpoint."""
    decoder = Decoder(embed_dim=ATTACK_CONFIG["embed_dim"]).to(device).eval()
    ckpt = torch.load(decoder_path, map_location="cpu")
    state = ckpt.get("decoder_state_dict", ckpt)
    cleaned = {k.removeprefix("module."): v for k, v in state.items()}
    decoder.load_state_dict(cleaned)
    return decoder


def generate_batch(clip_encoder, decoder, clean_images, target_images, eps):
    """Generate adversarial images for a batch."""
    with torch.no_grad():
        target_emb = clip_encoder.encode_img(target_images)
        noise = decoder(target_emb)
        noise = torch.clamp(noise, -eps, eps)
        adv_images = torch.clamp(clean_images + noise, 0, 1)
    return adv_images, noise


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_encoder = CLIPEncoder(ATTACK_CONFIG["clip_model"]).to(device)
    decoder = load_decoder(args.decoder_path, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.clean_image and args.target_image:
        # Single-pair mode
        clean = load_image(args.clean_image).to(device)
        target = load_image(args.target_image).to(device)
        adv, noise = generate_batch(clip_encoder, decoder, clean, target, args.eps)

        out_path = os.path.join(args.output_dir, "adversarial.png")
        torchvision.utils.save_image(adv[0], out_path)
        print(f"Saved: {out_path}")
        print(f"Noise L-inf: {noise.abs().max().item():.4f} "
              f"(budget: {args.eps:.4f})")
        print(f"PSNR: {-10 * torch.log10(torch.mean((clean - adv) ** 2)).item():.1f} dB")
        return

    # Batch mode
    if not (args.clean_dir and args.target_dir):
        print("Provide --clean-dir and --target-dir for batch generation.")
        return

    clean_loader = make_imagefolder_dataloader(
        args.clean_dir, args.batch_size, train=False
    )
    target_loader = make_imagefolder_dataloader(
        args.target_dir, args.batch_size, train=False
    )

    img_id = 0
    for (clean_batch, _), (target_batch, _) in tqdm(
        zip(clean_loader, target_loader), desc="Generating"
    ):
        clean_batch = clean_batch.to(device)
        target_batch = target_batch.to(device)
        adv_batch, _ = generate_batch(
            clip_encoder, decoder, clean_batch, target_batch, args.eps
        )
        for i in range(adv_batch.size(0)):
            path = os.path.join(args.output_dir, f"{img_id:05d}.png")
            torchvision.utils.save_image(adv_batch[i], path)
            img_id += 1

    print(f"Generated {img_id} adversarial images in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AnyAttack adversarial images")
    parser.add_argument("--decoder-path", type=str,
                        default=GENERATE_CONFIG["decoder_path"])
    parser.add_argument("--clean-image", type=str, default=None)
    parser.add_argument("--target-image", type=str, default=None)
    parser.add_argument("--clean-dir", type=str, default=None)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str,
                        default=GENERATE_CONFIG["output_dir"])
    parser.add_argument("--batch-size", type=int,
                        default=GENERATE_CONFIG["batch_size"])
    parser.add_argument("--eps", type=float, default=ATTACK_CONFIG["eps"])
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())
