"""图像处理工具函数（复用现有 demo 的模式）"""
import os
import logging
from datetime import datetime
from typing import Tuple
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境


def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """PIL → [1, 3, H, W] float tensor，值域 [0,1]"""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[1,3,H,W] 或 [3,H,W] → PIL Image"""
    if t.dim() == 4:
        t = t[0]
    arr = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_image(path: str, size: Tuple[int, int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BILINEAR)
    return img


def save_image(img, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(img, torch.Tensor):
        img = tensor_to_pil(img)
    img.save(path)


def calculate_psnr(orig: torch.Tensor, adv: torch.Tensor) -> float:
    mse = (orig.float() - adv.float()).pow(2).mean().item()
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def calculate_ssim(orig: torch.Tensor, adv: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        o = orig[0].permute(1, 2, 0).numpy()
        a = adv[0].permute(1, 2, 0).numpy()
        return float(ssim(o, a, data_range=1.0, channel_axis=2))
    except ImportError:
        return -1.0


def visualize_comparison(orig: Image.Image, adv: Image.Image,
                         save_path: str, title: str = ""):
    """保存原图 vs 对抗图的对比可视化"""
    orig_arr = np.array(orig)
    adv_arr  = np.array(adv)
    delta    = np.clip((adv_arr.astype(int) - orig_arr.astype(int) + 128), 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, label in zip(axes, [orig_arr, adv_arr, delta],
                               ["Original", "Adversarial", "Perturbation (×8)"]):
        ax.imshow(arr if label != "Perturbation (×8)"
                  else np.clip(delta.astype(int) * 8, 0, 255).astype(np.uint8))
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


def setup_run_dir(base_dir: str = "logs_and_outputs", prefix: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}" if prefix else ts
    run_dir = os.path.join(base_dir, name)
    for sub in ["adversarial", "visualizations"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir
