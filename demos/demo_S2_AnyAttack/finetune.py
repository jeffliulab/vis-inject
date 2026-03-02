"""
AnyAttack fine-tuning on COCO with auxiliary encoders.

Adapts the pre-trained Decoder to downstream tasks by optimizing against
multiple vision encoders (CLIP, EVA02-Large, ViT-B/16) simultaneously,
improving cross-model transferability.

Usage:
    python finetune.py
    python finetune.py --pretrain-checkpoint checkpoints/pre-trained.pt
    python finetune.py --criterion Cosine --epochs 10
"""

import argparse
import os
import sys
import time
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import FINETUNE_CONFIG, ATTACK_CONFIG
from models import CLIPEncoder, Decoder
from losses import BiContrastiveLoss, DirectMatchingLoss
from dataset import make_imagefolder_dataloader, TRAIN_TRANSFORM


def load_auxiliary_encoders(device):
    """Load frozen auxiliary encoders for cross-model transferability."""
    import timm
    import torchvision

    eva_encoder = timm.create_model(
        "hf_hub:timm/eva02_large_patch14_448.mim_m38m_ft_in1k",
        num_classes=0, pretrained=True
    ).to(device).eval()
    for p in eva_encoder.parameters():
        p.requires_grad = False

    imagenet_encoder = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    ).to(device).eval()
    imagenet_encoder.heads = torch.nn.Identity()
    for p in imagenet_encoder.parameters():
        p.requires_grad = False

    print(f"Auxiliary encoders loaded: EVA02-Large, ViT-B/16")
    return eva_encoder, imagenet_encoder


def load_coco_dataloader(data_dir: str, dataset_name: str, batch_size: int):
    """Load COCO dataset via LAVIS or fall back to ImageFolder."""
    try:
        from lavis.datasets.builders import load_dataset
        from lavis.common.registry import registry
        from dataset import coco_collate_fn

        registry.mapping["paths"]["cache_root"] = os.path.dirname(data_dir)
        coco = load_dataset(dataset_name, vis_path=data_dir)
        return DataLoader(
            coco["train"], batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, collate_fn=coco_collate_fn,
            drop_last=True,
        )
    except ImportError:
        print("[WARN] LAVIS not installed. Using data_dir as ImageFolder.")
        return make_imagefolder_dataloader(data_dir, batch_size, train=True)


def select_criterion(criterion_name: str):
    """Select loss function by name."""
    if criterion_name == "BiContrastiveLoss":
        return BiContrastiveLoss()
    elif criterion_name == "Cosine":
        return DirectMatchingLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_encoder = CLIPEncoder(ATTACK_CONFIG["clip_model"]).to(device)
    decoder = Decoder(embed_dim=ATTACK_CONFIG["embed_dim"]).to(device)

    # Load pre-trained weights
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        print(f"Loading pre-trained decoder: {args.pretrain_checkpoint}")
        ckpt = torch.load(args.pretrain_checkpoint, map_location=device)
        state = ckpt.get("decoder_state_dict", ckpt)
        # Handle DDP module. prefix
        cleaned = {}
        for k, v in state.items():
            cleaned[k.removeprefix("module.")] = v
        decoder.load_state_dict(cleaned)
    else:
        print("[WARN] No pre-trained checkpoint. Training from scratch.")

    # Auxiliary encoders
    eva_encoder, imagenet_encoder = (None, None)
    if args.use_auxiliary:
        eva_encoder, imagenet_encoder = load_auxiliary_encoders(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
    scaler = GradScaler()
    criterion = select_criterion(args.criterion)

    # Data loaders
    coco_loader = load_coco_dataloader(args.data_dir, args.dataset, args.batch_size)
    imagenet_loader = None
    imagenet_cycle = None
    if args.imagenet_dir and os.path.isdir(args.imagenet_dir):
        imagenet_loader = make_imagefolder_dataloader(
            args.imagenet_dir, args.batch_size, train=True
        )
        imagenet_cycle = cycle(imagenet_loader)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_sim_clip = 0.0
        count = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(coco_loader):
            if isinstance(batch, dict):
                images = batch["image"].to(device)
            else:
                images = batch[0].to(device)

            # Clean images from ImageNet (different from target images)
            if imagenet_cycle is not None:
                clean_batch = next(imagenet_cycle)
                clean_images = clean_batch[0].to(device)
            else:
                clean_images = images[torch.randperm(images.size(0))]

            with autocast():
                optimizer.zero_grad()

                with torch.no_grad():
                    embed_tar = clip_encoder.encode_img(images)

                noise = decoder(embed_tar)
                noise = torch.clamp(noise, -args.eps, args.eps)
                adv_images = torch.clamp(noise + clean_images, 0, 1)

                # CLIP loss
                embed_adv = clip_encoder.encode_img_with_grad(adv_images)
                loss_clip, sim_clip = criterion(embed_adv, embed_tar)
                total_loss_combined = loss_clip

                # Auxiliary encoder losses
                if eva_encoder is not None:
                    with torch.no_grad():
                        tar_eva = eva_encoder(
                            F.interpolate(images, size=(448, 448), mode="bilinear")
                        )
                    adv_eva = eva_encoder(
                        F.interpolate(adv_images, size=(448, 448), mode="bilinear")
                    )
                    loss_eva, _ = criterion(adv_eva, tar_eva)
                    total_loss_combined = total_loss_combined + loss_eva

                if imagenet_encoder is not None:
                    with torch.no_grad():
                        tar_in = imagenet_encoder(images)
                    adv_in = imagenet_encoder(adv_images)
                    loss_in, _ = criterion(adv_in, tar_in)
                    total_loss_combined = total_loss_combined + loss_in

            scaler.scale(total_loss_combined).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += total_loss_combined.item()
            total_sim_clip += sim_clip.item()
            count += 1
            scheduler.step()

            if batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss: {total_loss / count:.6f}, "
                      f"CLIP Sim: {total_sim_clip / count:.4f}, "
                      f"LR: {lr:.2e}")

        save_path = os.path.join(args.checkpoint_dir, "finetuned.pt")
        torch.save({
            "epoch": epoch + 1,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, save_path)
        elapsed = time.time() - t0
        print(f"Epoch {epoch} done in {elapsed:.0f}s. "
              f"Avg loss: {total_loss / max(count, 1):.6f}. Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnyAttack fine-tuning on COCO")
    cfg = FINETUNE_CONFIG
    parser.add_argument("--dataset", type=str, default=cfg["dataset"])
    parser.add_argument("--data-dir", type=str, default=cfg["data_dir"])
    parser.add_argument("--imagenet-dir", type=str, default=cfg["imagenet_dir"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--eps", type=float, default=ATTACK_CONFIG["eps"])
    parser.add_argument("--criterion", type=str, default=cfg["criterion"])
    parser.add_argument("--use-auxiliary", action="store_true",
                        default=cfg["use_auxiliary_encoders"])
    parser.add_argument("--no-auxiliary", action="store_false", dest="use_auxiliary")
    parser.add_argument("--pretrain-checkpoint", type=str,
                        default=cfg["pretrain_checkpoint"])
    parser.add_argument("--checkpoint-dir", type=str, default=cfg["checkpoint_dir"])
    main(parser.parse_args())
