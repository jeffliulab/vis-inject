"""
VisInject Utilities
====================
Shared functions for image loading, decoder loading, and metric computation.
"""

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _import_from_file(module_name: str, file_path: str):
    """Import a module from an explicit file path, avoiding sys.path pollution."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import AnyAttack components from demos via explicit paths
_s2_models_dir = os.path.join(os.path.dirname(__file__), "..", "demos", "demo_S2_AnyAttack", "models")
_clip_mod = _import_from_file("_anyattack_clip_encoder", os.path.join(_s2_models_dir, "clip_encoder.py"))
_decoder_mod = _import_from_file("_anyattack_decoder", os.path.join(_s2_models_dir, "decoder.py"))
CLIPEncoder = _clip_mod.CLIPEncoder
Decoder = _decoder_mod.Decoder


def load_image(image_path: str, size: int = 224) -> torch.Tensor:
    """Load image as (1, 3, H, W) tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def load_decoder(path: str, embed_dim: int = 512, device: torch.device = None) -> Decoder:
    """Load AnyAttack Decoder with state dict key remapping for official weights."""
    decoder = Decoder(embed_dim=embed_dim).to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("decoder_state_dict", ckpt)
    remapped = {}
    for k, v in state.items():
        k = k.removeprefix("module.")
        k = k.replace("upsample_blocks.", "blocks.")
        k = k.replace("final_conv.", "head.")
        remapped[k] = v
    decoder.load_state_dict(remapped)
    return decoder


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute PSNR between two image tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10 * torch.log10(torch.tensor(mse)).item()


def compute_clip_similarities(
    clip_encoder,
    images: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute pairwise CLIP cosine similarities between named images.

    Args:
        clip_encoder: CLIPEncoder instance.
        images: dict of name -> (1, 3, H, W) tensor.

    Returns:
        dict of "name1_vs_name2" -> cosine similarity.
    """
    embeddings = {}
    for name, img in images.items():
        emb = clip_encoder.encode_img(img)
        embeddings[name] = F.normalize(emb, p=2, dim=1)

    names = list(embeddings.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            sim = (embeddings[names[i]] * embeddings[names[j]]).sum().item()
            results[key] = sim
    return results
