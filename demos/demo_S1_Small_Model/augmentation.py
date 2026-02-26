"""
可微分图像增强层

两类用途：
A. 输入多样化（DI-FGSM 风格）：在计算编码器损失前随机变换图像，提升跨模型迁移性
B. 鲁棒性增强：模拟截图/JPEG/模糊等真实失真，训练扰动对这些操作的抵抗力
"""

import random
import math
import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# A. 输入多样化（提升跨模型迁移性）
# ============================================================

def input_diversity(images: Tensor, target_size: int,
                    prob: float = 0.5,
                    resize_range: tuple = (0.7, 1.0)) -> Tensor:
    """
    DI-FGSM 风格输入多样化：以 prob 的概率对图像做随机 resize + padding，
    最后 resize 到 target_size，保持梯度可传播。

    参数：
        images:      [B, C, H, W]，值域 [0,1]
        target_size: 目标尺寸（编码器期望的输入分辨率）
        prob:        施加多样化的概率
        resize_range: 随机 resize 的比例范围
    返回：
        [B, C, target_size, target_size]
    """
    if random.random() > prob:
        return F.interpolate(images, size=(target_size, target_size),
                             mode="bilinear", align_corners=False)

    # 随机选择中间尺寸
    scale = random.uniform(*resize_range)
    mid_size = max(1, int(target_size * scale))

    # 缩放到中间尺寸
    resized = F.interpolate(images, size=(mid_size, mid_size),
                            mode="bilinear", align_corners=False)

    # 计算 padding（使总尺寸 = target_size）
    pad_total = target_size - mid_size
    pad_top   = random.randint(0, pad_total)
    pad_left  = random.randint(0, pad_total)
    pad_bot   = pad_total - pad_top
    pad_right = pad_total - pad_left

    # F.pad: (left, right, top, bottom)
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bot),
                   mode="constant", value=0.5)

    # 确保输出尺寸精确为 target_size
    if padded.shape[-1] != target_size or padded.shape[-2] != target_size:
        padded = F.interpolate(padded, size=(target_size, target_size),
                               mode="bilinear", align_corners=False)

    return padded


# ============================================================
# B. 鲁棒性增强（模拟真实失真）
# ============================================================

def quantize_ste(images: Tensor, bits: int = 8) -> Tensor:
    """
    色彩量化（Straight-Through Estimator）。
    前向：将像素量化到 2^bits 个灰度级（模拟 PNG uint8 存储）
    反向：梯度直通，不受量化截断影响
    复用自 demo_C1_CLIP_ViT 的量化感知攻击思路。
    """
    levels = 2 ** bits
    quantized = (images * levels).round() / levels
    return images + (quantized - images).detach()


def jpeg_like_compression(images: Tensor, quality: int = 50,
                          patch_size: int = 8) -> Tensor:
    """
    简化的可微分 JPEG 量化模拟（DCT 系数截断）。
    quality 越低，量化步长越大，压缩损失越大。
    用 STE 保持梯度可传播。

    注意：这是近似实现，不完全等同于标准 JPEG，但足以训练鲁棒性。
    """
    from models.stego_encoder import dct2d_blockwise, idct2d_blockwise, _get_dct_matrix
    import math

    B, C, H, W = images.shape

    # 量化矩阵（亮度量化表近似，quality 越低步长越大）
    quality = max(1, min(99, quality))
    scale = (100 - quality) / 50.0 if quality >= 50 else 50.0 / quality

    # 8×8 标准亮度量化矩阵（JPEG 标准）
    std_luma = torch.tensor([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99],
    ], dtype=images.dtype, device=images.device)
    Q = torch.clamp(std_luma * scale, 1, 255).view(1, 1, patch_size, patch_size)

    # 1. [0,1] → [-128, 127]（JPEG 中心化）
    shifted = images * 255 - 128

    # 2. 分块 DCT
    dct = dct2d_blockwise(shifted, patch_size)

    # 3. 量化（STE）
    ps = patch_size
    nH = (H + ps - 1) // ps
    nW = (W + ps - 1) // ps
    Q_tiled = Q.repeat(1, 1, nH, nW)[:, :, :H, :W]
    dct_q = (dct / Q_tiled).round() * Q_tiled
    dct_compressed = dct + (dct_q - dct).detach()   # STE

    # 4. IDCT + 还原
    reconstructed = idct2d_blockwise(dct_compressed, patch_size)
    output = torch.clamp((reconstructed + 128) / 255.0, 0.0, 1.0)

    return output


def gaussian_blur(images: Tensor, sigma: float = None,
                  kernel_size: int = None) -> Tensor:
    """
    可微分高斯模糊（模拟截图/显示屏采样效果）。
    sigma 随机选择以覆盖轻微到中度模糊。
    """
    if sigma is None:
        sigma = random.uniform(0.5, 2.0)
    if kernel_size is None:
        kernel_size = max(3, int(4 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

    # 构建可分离高斯核
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1,
                     dtype=images.dtype, device=images.device)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    C = images.shape[1]
    kernel_2d = gauss.outer(gauss).view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    blurred = F.conv2d(
        F.pad(images, [pad, pad, pad, pad], mode="reflect"),
        kernel_2d, groups=C
    )
    return blurred


def random_crop_and_pad(images: Tensor, crop_ratio: float = None) -> Tensor:
    """
    随机裁去图像边缘后 pad 回原尺寸（模拟用户截取局部图像）。
    crop_ratio: 裁去比例，如 0.1 表示裁去各方向 10%
    """
    if crop_ratio is None:
        crop_ratio = random.uniform(0.05, 0.15)

    B, C, H, W = images.shape
    ch = int(H * crop_ratio)
    cw = int(W * crop_ratio)
    # 随机偏移（不总是从四边均匀裁）
    top  = random.randint(0, max(0, ch))
    left = random.randint(0, max(0, cw))
    bot  = random.randint(0, max(0, ch))
    right = random.randint(0, max(0, cw))

    top_e  = H - top - bot
    left_e = W - left - right
    if top_e <= 0 or left_e <= 0:
        return images

    cropped = images[:, :, top:top + top_e, left:left + left_e]
    # 还原到原始尺寸
    restored = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
    return restored


def scale_and_restore(images: Tensor, scale: float = None) -> Tensor:
    """
    缩放后还原（模拟分辨率变化对图像的影响）。
    scale < 1: 降采样再上采样（损失高频细节）
    scale > 1: 上采样再降采样（轻微平滑）
    """
    if scale is None:
        scale = random.choice([0.5, 0.75, 1.5, 2.0])

    B, C, H, W = images.shape
    new_h = max(1, int(H * scale))
    new_w = max(1, int(W * scale))

    scaled = F.interpolate(images, size=(new_h, new_w),
                           mode="bilinear", align_corners=False)
    restored = F.interpolate(scaled, size=(H, W),
                             mode="bilinear", align_corners=False)
    return restored


def screenshot_simulate(images: Tensor) -> Tensor:
    """
    截图模拟：先用双线性降采样再上采样，加轻微高斯模糊，再量化。
    模拟在屏幕上显示图像然后截图的过程。
    """
    # 1. 双线性降采样再上采样（模拟显示器像素网格采样）
    B, C, H, W = images.shape
    down_scale = random.uniform(0.8, 0.95)
    down_h = max(1, int(H * down_scale))
    down_w = max(1, int(W * down_scale))
    images = F.interpolate(images, size=(down_h, down_w),
                           mode="bilinear", align_corners=False)
    images = F.interpolate(images, size=(H, W),
                           mode="bilinear", align_corners=False)
    # 2. 轻微模糊（显示器的 anti-aliasing）
    images = gaussian_blur(images, sigma=random.uniform(0.3, 0.8))
    # 3. 量化（截图存为 PNG uint8）
    images = quantize_ste(images, bits=8)
    return images


# ============================================================
# 组合接口
# ============================================================

class DifferentiableAugmentor:
    """
    统一增强接口，在训练和评估中使用。
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}

    def diversity_aug(self, images: Tensor, target_size: int,
                      prob: float = 0.5) -> Tensor:
        """输入多样化（A 类）：在计算编码器损失前调用"""
        return input_diversity(images, target_size, prob=prob)

    def robustness_aug(self, images: Tensor,
                       distortion: str = "random") -> Tensor:
        """
        鲁棒性增强（B 类）：模拟真实失真，在 RL 和评估中使用。
        distortion: "random" | "jpeg_q50" | "jpeg_q30" | "scale_half" |
                    "scale_double" | "gaussian_blur" | "screenshot_sim" | "none"
        """
        if distortion == "none":
            return images

        if distortion == "random":
            distortion = random.choice([
                "jpeg_q50", "jpeg_q30", "scale_half",
                "gaussian_blur", "screenshot_sim", "none"
            ])

        if distortion == "jpeg_q50":
            return jpeg_like_compression(images, quality=50)
        elif distortion == "jpeg_q30":
            return jpeg_like_compression(images, quality=30)
        elif distortion == "scale_half":
            return scale_and_restore(images, scale=0.5)
        elif distortion == "scale_double":
            return scale_and_restore(images, scale=2.0)
        elif distortion == "gaussian_blur":
            return gaussian_blur(images, sigma=random.uniform(1.0, 2.0))
        elif distortion == "screenshot_sim":
            return screenshot_simulate(images)
        elif distortion == "crop":
            return random_crop_and_pad(images)
        elif distortion == "quantize":
            return quantize_ste(images)
        else:
            return images

    def training_aug(self, images: Tensor) -> Tensor:
        """
        训练时的混合增强：随机选择一种失真。
        用于 proxy_trainer 的每个 batch，在计算多编码器损失后还用于鲁棒性正则。
        """
        return self.robustness_aug(images, distortion="random")
