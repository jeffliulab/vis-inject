"""
StegoEncoder — 核心图像变换小模型
架构：U-Net 残差 CNN + DCT 中频域约束 + 低通梯度平滑 + FiLM 文本条件（可选）

关键设计：
1. StegoEncoder 输出的扰动在 DCT 中频域（band 3-15），而非像素空间高频噪声
   → 这是实现跨模型迁移性的核心：所有 ViT 都必须处理中频 DCT 信号
2. 每个训练 step 后对扰动做低通平滑（去除反传引入的高频）
3. FiLM 条件接口预留，默认关闭（模式 A：固定 Token）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# 基础构建块
# ============================================================

class ResBlock(nn.Module):
    """带 GroupNorm 的残差块（FiLM 条件注入点）"""

    def __init__(self, channels: int, film_dim: int = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act   = nn.SiLU()

        # FiLM 条件参数（若 film_dim > 0 则激活）
        self.use_film = film_dim > 0
        if self.use_film:
            self.film_linear = nn.Linear(film_dim, channels * 2)

    def forward(self, x: torch.Tensor,
                film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))

        # FiLM 调制：scale/shift 归一化输出
        if self.use_film and film_cond is not None:
            gamma_beta = self.film_linear(film_cond)          # [B, 2C]
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0   # 初始化为 1
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            h = h * gamma + beta

        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_ch: int, out_ch: int, film_dim: int = 0, n_res: int = 2):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_ch if i == 0 else out_ch, film_dim) for i in range(n_res)]
        )
        # 修正：第一个 res block 在通道不一致时需要投影
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, film_dim) for _ in range(n_res)])
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, film_cond=None):
        x = self.proj_in(x)
        for blk in self.res_blocks:
            x = blk(x, film_cond)
        return self.down(x), x  # (下采样后, skip connection)


class UpBlock(nn.Module):
    """上采样块（接收 skip connection）"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 film_dim: int = 0, n_res: int = 2):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, film_dim) for _ in range(n_res)])

    def forward(self, x, skip, film_cond=None):
        x = self.up(x)
        # 处理尺寸不匹配（奇数分辨率时）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(torch.cat([x, skip], dim=1))
        for blk in self.res_blocks:
            x = blk(x, film_cond)
        return x


# ============================================================
# DCT 工具函数
# ============================================================

def _get_dct_matrix(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    生成 N×N 正交 DCT-II 变换矩阵 D，满足 D @ D.T = I。
    D[k, n] = w_k * cos(pi * k * (n + 0.5) / N)
    w_0 = 1/sqrt(N),  w_k = sqrt(2/N)  for k > 0
    """
    n_vec = torch.arange(N, dtype=dtype, device=device)   # [0..N-1]
    k_vec = torch.arange(N, dtype=dtype, device=device)   # [0..N-1]
    # 余弦部分：D[k, n] = cos(pi * k * (n + 0.5) / N)
    D = torch.cos(torch.pi * k_vec.unsqueeze(1)       # [N, 1]
                  * (n_vec.unsqueeze(0) + 0.5)        # [1, N]
                  / N)                                # [N, N]
    # 逐行归一化：w_0 = 1/sqrt(N), w_k = sqrt(2/N)
    D[0]  = D[0]  * (1.0 / N ** 0.5)
    D[1:] = D[1:] * ((2.0 / N) ** 0.5)
    return D   # [N, N]


def dct2d_blockwise(x: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    分块 2D DCT-II（正交）。
    输入：[B, C, H, W]  值域任意
    输出：[B, C, H, W]  DCT 系数图（与输入同尺寸）
    变换：X_block = D @ block @ D.T
    """
    B, C, H, W = x.shape
    ps = patch_size
    pad_h = (ps - H % ps) % ps
    pad_w = (ps - W % ps) % ps
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H2, W2 = x.shape

    D = _get_dct_matrix(ps, x.device, x.dtype)   # [ps, ps]
    nH, nW = H2 // ps, W2 // ps

    # 提取 patches: [B, C, nH, nW, ps, ps]
    patches = (x.unfold(2, ps, ps)               # [B, C, nH, W2, ps]
                .unfold(3, ps, ps)               # [B, C, nH, nW, ps, ps]
                .contiguous()
                .view(-1, ps, ps))               # [B*C*nH*nW, ps, ps]

    # 2D DCT: X = D @ patch @ D.T
    dct_patches = D @ patches @ D.T              # [N, ps, ps]

    # 重组为图像
    dct_patches = dct_patches.view(B, C, nH, nW, ps, ps)
    dct_out = (dct_patches
               .permute(0, 1, 2, 4, 3, 5)       # [B, C, nH, ps, nW, ps]
               .contiguous()
               .view(B, C, H2, W2))

    return dct_out[:, :, :H, :W]


def idct2d_blockwise(dct_x: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    分块 2D 逆 DCT（IDCT-II = DCT-III / N = D.T @ X @ D）。
    输入：[B, C, H, W] DCT 系数图
    输出：[B, C, H, W] 重建图像
    """
    B, C, H, W = dct_x.shape
    ps = patch_size
    pad_h = (ps - H % ps) % ps
    pad_w = (ps - W % ps) % ps
    if pad_h > 0 or pad_w > 0:
        dct_x = F.pad(dct_x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H2, W2 = dct_x.shape

    D = _get_dct_matrix(ps, dct_x.device, dct_x.dtype)   # [ps, ps]
    nH, nW = H2 // ps, W2 // ps

    patches = (dct_x.unfold(2, ps, ps)
                     .unfold(3, ps, ps)
                     .contiguous()
                     .view(-1, ps, ps))          # [N, ps, ps]

    # IDCT: x = D.T @ X @ D
    img_patches = D.T @ patches @ D

    img_patches = img_patches.view(B, C, nH, nW, ps, ps)
    img_out = (img_patches
               .permute(0, 1, 2, 4, 3, 5)
               .contiguous()
               .view(B, C, H2, W2))

    return img_out[:, :, :H, :W]


def make_mid_freq_mask(patch_size: int, band_low: int, band_high: int,
                       device: torch.device) -> torch.Tensor:
    """
    生成中频 mask：对每个 patch_size×patch_size 的 DCT block，
    只保留 zigzag 序号在 [band_low, band_high] 之间的系数为 1，其余为 0。
    输出：[1, 1, patch_size, patch_size] 可广播
    """
    mask = torch.zeros(patch_size, patch_size, device=device)
    idx = 0
    # zigzag 扫描顺序
    for s in range(2 * patch_size - 1):
        if s % 2 == 0:
            r_start = min(s, patch_size - 1)
            c_start = s - r_start
            while r_start >= 0 and c_start < patch_size:
                if band_low <= idx <= band_high:
                    mask[r_start, c_start] = 1.0
                idx += 1
                r_start -= 1
                c_start += 1
        else:
            c_start = min(s, patch_size - 1)
            r_start = s - c_start
            while c_start >= 0 and r_start < patch_size:
                if band_low <= idx <= band_high:
                    mask[r_start, c_start] = 1.0
                idx += 1
                c_start -= 1
                r_start += 1
    return mask.view(1, 1, patch_size, patch_size)


def gaussian_lowpass_2d(delta: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """对扰动图像做高斯低通平滑（去除高频成分），使用可分离卷积"""
    # 构建高斯核
    kernel_size = max(3, int(4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=delta.device, dtype=delta.dtype)
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d.outer(gauss_1d)                         # [k, k]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(
        delta.shape[1], 1, kernel_size, kernel_size
    )   # [C, 1, k, k]

    pad = half
    smoothed = F.conv2d(
        F.pad(delta, [pad, pad, pad, pad], mode="reflect"),
        kernel, groups=delta.shape[1]
    )
    return smoothed


# ============================================================
# 主模型
# ============================================================

class StegoEncoder(nn.Module):
    """
    图像隐写注入网络。

    模式 A（fixed_token，默认）：
        输入：原始图像 [B, 3, H, W]
        输出：对抗图像 [B, 3, H, W]（中频扰动，L∞ ≤ epsilon）

    模式 B（controllable，扩展）：
        额外输入：文本条件向量 film_cond [B, text_embed_dim]
        通过 FiLM 条件调制 U-Net 中间层，实现可控 prompt 注入
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg       = cfg
        self.epsilon   = cfg.get("epsilon", 16 / 255)
        self.patch_size = cfg.get("dct_patch_size", 8)
        self.band_low  = cfg.get("freq_band_low", 3)
        self.band_high = cfg.get("freq_band_high", 15)
        self.sigma     = cfg.get("lowpass_sigma", 1.0)
        self.use_film  = cfg.get("use_film_conditioning", False)
        film_dim = cfg.get("text_embed_dim", 512) if self.use_film else 0

        C  = cfg.get("base_channels", 64)
        nr = cfg.get("num_res_blocks", 4)

        # ---- U-Net 编码器路径（3 个尺度） ----
        self.stem    = nn.Conv2d(3, C, 3, padding=1)
        self.down1   = DownBlock(C,     C * 2, film_dim, nr)
        self.down2   = DownBlock(C * 2, C * 4, film_dim, nr)
        self.down3   = DownBlock(C * 4, C * 8, film_dim, nr)

        # ---- 瓶颈层 ----
        self.bottleneck = nn.Sequential(
            *[ResBlock(C * 8, film_dim) for _ in range(nr)]
        )

        # ---- U-Net 解码器路径 ----
        # skip_ch 必须与对应 DownBlock 的 out_ch 一致（skip 在 down 输出、下采样之前取）
        self.up3 = UpBlock(C * 8, C * 8, C * 4, film_dim, nr)   # skip from down3 (C*8)
        self.up2 = UpBlock(C * 4, C * 4, C * 2, film_dim, nr)   # skip from down2 (C*4)
        self.up1 = UpBlock(C * 2, C * 2, C,     film_dim, nr)   # skip from down1 (C*2)

        # ---- 输出头：生成 DCT 系数修改量 ----
        self.head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(C, 3, 1),    # 输出与图像同通道
        )

        # mid-freq mask（注册为 buffer，跟随 device）
        self.register_buffer(
            "freq_mask",
            make_mid_freq_mask(self.patch_size, self.band_low, self.band_high,
                               device=torch.device("cpu"))
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _unet_forward(self, x: torch.Tensor,
                      film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """U-Net 前向传播，返回原始输出（未约束）"""
        s0 = self.stem(x)
        s1, skip1 = self.down1(s0, film_cond)
        s2, skip2 = self.down2(s1, film_cond)
        s3, skip3 = self.down3(s2, film_cond)

        # 瓶颈：ResBlock 不接受 film_cond（Sequential），手动循环
        h = s3
        for blk in self.bottleneck:
            h = blk(h, film_cond)

        h = self.up3(h, skip3, film_cond)
        h = self.up2(h, skip2, film_cond)
        h = self.up1(h, skip1, film_cond)
        return self.head(h)

    def apply_dct_constraint(self, orig: torch.Tensor,
                             raw_delta: torch.Tensor) -> torch.Tensor:
        """
        将 U-Net 输出的 raw_delta 约束到 DCT 中频域，然后转回像素空间。
        """
        H, W = orig.shape[-2], orig.shape[-1]

        # 对原始图像做 DCT
        orig_dct = dct2d_blockwise(orig, self.patch_size)

        # 对 raw_delta 做 DCT，只保留中频部分
        # freq_mask 需要按 patch 位置广播
        # 方法：在 DCT 域直接按 patch 位置的 mask tile
        ps = self.patch_size
        nH = (H + ps - 1) // ps
        nW = (W + ps - 1) // ps
        # 将 mask tile 到整图大小
        mask_tiled = self.freq_mask.repeat(1, 1, nH, nW)[:, :, :H, :W]  # [1,1,H,W]

        # U-Net 输出本身是像素域，转到 DCT 域再 mask
        raw_delta_dct = dct2d_blockwise(raw_delta, self.patch_size)
        constrained_dct = raw_delta_dct * mask_tiled  # 只保留中频系数

        # 加到原图 DCT 上，再做 IDCT 得到扰动
        modified_dct = orig_dct + constrained_dct
        adv_img = idct2d_blockwise(modified_dct, self.patch_size)

        return adv_img

    def forward(self, images: torch.Tensor,
                film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        主前向传播。
        输入：images [B, 3, H, W]，值域 [0, 1]
        输出：adv_images [B, 3, H, W]，值域 [0, 1]，扰动 L∞ ≤ epsilon
        """
        # U-Net 输出
        raw_delta = self.head(self._run_unet(images, film_cond))

        # DCT 中频约束
        adv_img = self.apply_dct_constraint(images, raw_delta)

        # L∞ 约束：扰动幅度不超过 epsilon
        delta = adv_img - images
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        adv_img = torch.clamp(images + delta, 0.0, 1.0)

        return adv_img

    def _run_unet(self, x: torch.Tensor,
                  film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回 U-Net 解码后的特征图（head 之前），由 forward 调用 self.head 输出"""
        s0 = self.stem(x)
        s1, skip1 = self.down1(s0, film_cond)
        s2, skip2 = self.down2(s1, film_cond)
        s3, skip3 = self.down3(s2, film_cond)
        h = s3
        for blk in self.bottleneck:
            h = blk(h, film_cond)
        h = self.up3(h, skip3, film_cond)
        h = self.up2(h, skip2, film_cond)
        h = self.up1(h, skip1, film_cond)
        return h  # 注意：不调用 head，由 forward 调用

    @torch.no_grad()
    def apply_lowpass_smoothing(self, adv_img: torch.Tensor,
                                orig: torch.Tensor) -> torch.Tensor:
        """
        训练 step 后对扰动做低通平滑，去除反传引入的高频伪影。
        应在 optimizer.step() 之后、下一个 batch 之前调用。
        """
        delta = adv_img - orig
        delta_smooth = gaussian_lowpass_2d(delta, self.sigma)
        delta_smooth = torch.clamp(delta_smooth, -self.epsilon, self.epsilon)
        return torch.clamp(orig + delta_smooth, 0.0, 1.0)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
