"""
多编码器集成代理损失

自动遍历 encoders/ 注册表中已加载的编码器计算特征损失，
新增编码器无需修改此文件。

总损失 = Σᵢ λᵢ · L_encoder_i + λ_percept · L_percept + λ_distort · L_distort + λ_freq · L_freq_reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch import Tensor


# ============================================================
# 感知损失（VGG）
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """
    基于 VGG-16 relu3_3 特征的感知损失。
    用于保持对抗图像与原图的语义相似性。
    """

    def __init__(self):
        super().__init__()
        self._vgg = None

    def _load_vgg(self, device):
        if self._vgg is None:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._vgg = nn.Sequential(*list(vgg.features)[:16]).to(device).eval()
            for p in self._vgg.parameters():
                p.requires_grad_(False)

    def forward(self, adv_img: Tensor, orig_img: Tensor) -> Tensor:
        self._load_vgg(adv_img.device)
        # VGG 期望 ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=adv_img.device, dtype=adv_img.dtype).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                            device=adv_img.device, dtype=adv_img.dtype).view(1, 3, 1, 1)

        # 统一 resize 到 VGG 期望尺寸（避免超大图 OOM）
        def prep(x):
            if x.shape[-1] > 256:
                x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            return (x - mean) / std

        f_adv  = self._vgg(prep(adv_img))
        f_orig = self._vgg(prep(orig_img.detach()))
        return F.mse_loss(f_adv, f_orig)


# ============================================================
# 频率正则损失（惩罚高频扰动）
# ============================================================

def freq_reg_loss(adv_img: Tensor, orig_img: Tensor,
                  patch_size: int = 8, high_band_start: int = 20) -> Tensor:
    """
    惩罚扰动中高频 DCT 系数的能量。
    迫使 StegoEncoder 的扰动集中在中低频，提升跨模型迁移性。
    """
    from models.stego_encoder import dct2d_blockwise, make_mid_freq_mask
    ps = patch_size
    delta = adv_img - orig_img.detach()
    dct   = dct2d_blockwise(delta, ps)

    # 高频 mask（band >= high_band_start）
    H, W = delta.shape[-2], delta.shape[-1]
    # 简化：直接惩罚所有 DCT 系数绝对值（高频系数在 mask 之外的部分）
    low_mask = make_mid_freq_mask(ps, band_low=0, band_high=high_band_start - 1,
                                  device=dct.device)
    nH = (H + ps - 1) // ps
    nW = (W + ps - 1) // ps
    low_mask_tiled = low_mask.repeat(1, 1, nH, nW)[:, :, :H, :W]
    high_energy = (dct * (1.0 - low_mask_tiled)).pow(2).mean()
    return high_energy


# ============================================================
# 编码器特征对齐损失（Oracle 特征匹配）
# ============================================================

def cosine_feature_loss(feat_adv: Tensor, feat_target: Tensor) -> Tensor:
    """
    1 - cosine_similarity(feat_adv, feat_target)，在 feature dim 上计算。
    feat_adv, feat_target: [B, D]
    """
    cos_sim = F.cosine_similarity(feat_adv, feat_target.detach(), dim=-1)
    return (1.0 - cos_sim).mean()


# ============================================================
# 主损失函数
# ============================================================

class MultiEncoderProxyLoss(nn.Module):
    """
    多编码器集成代理损失。
    自动遍历传入的编码器列表，无需手写每个模型的损失分支。
    新增编码器：只需将其加入 config.ACTIVE_ENCODERS，此处无需修改。
    """

    def __init__(self, weights: dict, use_perceptual: bool = True):
        """
        weights: config.LOSS_WEIGHTS，含 encoder/percept/distort/freq_reg 权重
        """
        super().__init__()
        self.w = weights
        self.use_perceptual = use_perceptual
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    def forward(
        self,
        adv_img: Tensor,
        orig_img: Tensor,
        encoders: list,
        oracle_features: Dict[str, Tensor],
        augmentor=None,
    ) -> tuple:
        """
        参数：
            adv_img:        [B, 3, H, W]  对抗图像（值域 [0,1]）
            orig_img:       [B, 3, H, W]  原始图像（值域 [0,1]）
            encoders:       List[BaseVisualEncoder]，已加载的编码器列表
            oracle_features: {enc_name: [B, D]}  各编码器的目标特征向量
            augmentor:      DifferentiableAugmentor 或 None

        返回：
            (total_loss, loss_dict)
            loss_dict: {"encoder_blip2": ..., "percept": ..., "distort": ..., ...}
        """
        loss_dict = {}

        # ---- 1. 多编码器特征对齐损失 ----
        enc_losses = []
        for enc in encoders:
            if enc.name not in oracle_features:
                continue
            oracle_feat = oracle_features[enc.name]   # [B, D]

            # 输入多样化（DI-FGSM）：在编码器尺寸下 resize + 随机变换
            if augmentor is not None:
                aug_img = augmentor.diversity_aug(adv_img, enc.img_size, prob=0.5)
            else:
                aug_img = F.interpolate(adv_img,
                                        size=(enc.img_size, enc.img_size),
                                        mode="bilinear", align_corners=False)

            # 编码器专用归一化
            normed = enc.normalize(aug_img)

            # 提取特征（编码器已冻结，但需要梯度流过去用于 proxy loss）
            feat_adv = enc.encode(normed)   # [B, D] 或 [B, N, D]
            if feat_adv.dim() == 3:
                feat_adv = feat_adv.mean(dim=1)

            l_enc = cosine_feature_loss(feat_adv, oracle_feat)
            enc_losses.append(enc.weight * l_enc)
            loss_dict[f"encoder_{enc.name}"] = l_enc.item()

        enc_total = torch.stack(enc_losses).sum() if enc_losses else torch.tensor(0.0)
        loss_dict["encoder_total"] = enc_total.item()

        # ---- 2. 感知损失（保持语义不变） ----
        if self.use_perceptual and self.perceptual is not None:
            l_percept = self.perceptual(adv_img, orig_img)
        else:
            l_percept = torch.tensor(0.0, device=adv_img.device)
        loss_dict["percept"] = l_percept.item()

        # ---- 3. 失真惩罚（L2） ----
        l_distort = F.mse_loss(adv_img, orig_img.detach())
        loss_dict["distort"] = l_distort.item()

        # ---- 4. 频率正则（惩罚高频扰动） ----
        l_freq = freq_reg_loss(adv_img, orig_img)
        loss_dict["freq_reg"] = l_freq.item()

        # ---- 加权求和 ----
        total = (self.w.get("encoder", 1.0) * enc_total
               + self.w.get("percept",  0.1) * l_percept
               + self.w.get("distort",  0.5) * l_distort
               + self.w.get("freq_reg", 0.2) * l_freq)

        loss_dict["total"] = total.item()
        return total, loss_dict


# ============================================================
# Oracle 特征预计算工具
# ============================================================

def compute_oracle_features_pgd(
    orig_imgs: Tensor,
    encoders: list,
    pgd_steps: int = 20,
    pgd_alpha: float = 1 / 255,
    pgd_eps: float = 16 / 255,
) -> Dict[str, Tensor]:
    """
    为当前 batch 的图像，用轻量 PGD 生成 oracle 攻击图，
    然后提取各编码器的特征作为训练目标。

    动态 oracle（每 batch 生成）避免了天量预计算的需求。
    pgd_steps 较少（20步），生成速度快，质量够用。
    """
    oracle_features = {}

    for enc in encoders:
        if not enc.is_loaded():
            continue

        # 初始化扰动
        delta = torch.zeros_like(orig_imgs).uniform_(-pgd_eps, pgd_eps)
        orig_feat = enc.resize_and_encode(orig_imgs.detach())

        for _ in range(pgd_steps):
            delta.requires_grad_(True)
            adv = torch.clamp(orig_imgs + delta, 0, 1)
            feat = enc.resize_and_encode(adv)

            # 最大化与原始特征的距离（朝"不同于原始"的方向移动）
            loss = -F.cosine_similarity(feat, orig_feat.detach(), dim=-1).mean()
            loss.backward()

            with torch.no_grad():
                delta = delta - pgd_alpha * delta.grad.sign()
                delta = torch.clamp(delta, -pgd_eps, pgd_eps)
            delta = delta.detach()

        with torch.no_grad():
            oracle_adv  = torch.clamp(orig_imgs + delta, 0, 1)
            oracle_feat = enc.resize_and_encode(oracle_adv)
        oracle_features[enc.name] = oracle_feat.detach()

    return oracle_features
