"""
Stage 1 代理预训练循环

Stage 1A：单编码器，少量图像，4090 快速验证（~1-2 小时）
Stage 1B：全编码器，5k 图像，多编码器集成损失（~4-6 小时）
Stage 1C：HPC 端到端全模型训练（扩展接口，此文件不含 LLM 梯度）

通过 config.ACTIVE_ENCODERS 和 STAGE1A/1B_CONFIG 切换两种模式。
"""

import os
import json
import logging
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

import config as cfg
from models.stego_encoder import StegoEncoder
from losses import MultiEncoderProxyLoss, compute_oracle_features_pgd
from augmentation import DifferentiableAugmentor


# ============================================================
# 工具函数
# ============================================================

def setup_logging(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, "train.log")
    logger = logging.getLogger("proxy_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_image_paths(data_dir: str, num_images: int,
                     extensions=(".jpg", ".jpeg", ".png")) -> List[str]:
    """从目录加载图像路径，若 data_dir 不存在则返回随机噪声标记（用于 dry run）"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return [f"__synthetic_{i}__" for i in range(num_images)]

    paths = [str(p) for p in data_dir.iterdir()
             if p.suffix.lower() in extensions]
    random.shuffle(paths)
    return paths[:num_images]


def load_image_tensor(path: str, device: str,
                      size: int = 256) -> Optional[torch.Tensor]:
    """
    加载单张图像为张量 [1, 3, H, W]，值域 [0,1]。
    使用中心裁剪而非直接拉伸，保持图像内容的宽高比。
    """
    if path.startswith("__synthetic_"):
        return torch.rand(1, 3, size, size, device=device)
    try:
        img = Image.open(path).convert("RGB")
        # 保持比例：缩放短边到 size，再中心裁剪
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - size) // 2
        top  = (new_h - size) // 2
        img  = img.crop((left, top, left + size, top + size))

        arr = __import__("numpy").array(img, dtype=__import__("numpy").float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(device)
    except Exception:
        return None


def compute_psnr(orig: torch.Tensor, adv: torch.Tensor) -> float:
    mse = (orig - adv).pow(2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


# ============================================================
# 主训练类
# ============================================================

class ProxyTrainer:
    """
    Stage 1 代理预训练。
    通过 stage_cfg 参数区分 Stage 1A / 1B，训练逻辑完全一致。
    """

    def __init__(self, stage: str = "1a"):
        assert stage in ("1a", "1b", "1c"), f"无效 stage: {stage}"
        self.stage = stage
        self.stage_cfg = cfg.STAGE1A_CONFIG if stage == "1a" else cfg.STAGE1B_CONFIG
        self.device = cfg.DEVICE

        # 输出目录
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("logs_and_outputs", f"stage{stage}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        self.logger = setup_logging(self.run_dir)
        self.logger.info(f"Stage {stage.upper()} 训练初始化")
        self.logger.info(f"输出目录: {self.run_dir}")

    def _load_encoders(self) -> list:
        from encoders import load_encoders
        # Stage 1A 只用第一个编码器（如果列表非空），1B 用全部
        enc_names = (cfg.ACTIVE_ENCODERS if self.stage != "1a" or not cfg.ACTIVE_ENCODERS
                     else [cfg.ACTIVE_ENCODERS[0]])
        self.logger.info(f"加载编码器: {enc_names}")
        encoders = load_encoders(enc_names, cfg.ENCODER_CONFIG)
        for enc in encoders:
            self.logger.info(f"  加载 {enc.name} ...")
            enc.load()
            self.logger.info(f"  {enc.name} 加载完成")
        return encoders

    def run(self):
        """执行完整的 Stage 1 训练循环"""
        self.logger.info("=" * 60)
        self.logger.info(f"开始 Stage {self.stage.upper()} 代理预训练")
        self.logger.info(f"  编码器: {cfg.ACTIVE_ENCODERS}")
        self.logger.info(f"  Prompt: {cfg.ACTIVE_PROMPT}")
        self.logger.info(f"  图像数量: {self.stage_cfg['num_images']}")
        self.logger.info(f"  Epochs: {self.stage_cfg['epochs']}")
        self.logger.info("=" * 60)

        # ---- 初始化 ----
        torch.manual_seed(cfg.SEED)
        encoders = self._load_encoders()

        model = StegoEncoder(cfg.STEGO_MODEL_CONFIG).to(self.device)
        self.logger.info(f"StegoEncoder 参数量: {model.parameter_count()/1e6:.2f}M")

        optimizer = optim.AdamW(model.parameters(),
                                lr=self.stage_cfg["lr"],
                                weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.stage_cfg["epochs"], eta_min=1e-6
        )

        loss_fn  = MultiEncoderProxyLoss(cfg.LOSS_WEIGHTS, use_perceptual=False)
        augmentor = DifferentiableAugmentor()

        # ---- 数据 ----
        # 使用编码器的 img_size 加载图像（Qwen=392，需是 patch_size*merge_size=28 的倍数）
        img_size = encoders[0].img_size if encoders else 256
        self.logger.info(f"图像分辨率: {img_size}×{img_size}")

        image_paths = load_image_paths("data/train", self.stage_cfg["num_images"])
        self.logger.info(f"图像数量: {len(image_paths)}")

        # ---- 训练循环 ----
        history = []
        best_loss = float("inf")

        for epoch in range(1, self.stage_cfg["epochs"] + 1):
            model.train()
            epoch_losses = []
            random.shuffle(image_paths)

            batch_size = self.stage_cfg.get("batch_size", 4)
            batches = [image_paths[i:i+batch_size]
                       for i in range(0, len(image_paths), batch_size)]

            epoch_bar = tqdm(batches, desc=f"Epoch {epoch}/{self.stage_cfg['epochs']}",
                             leave=False)

            for batch_paths in epoch_bar:
                # 加载 batch 图像（按编码器 img_size 加载，保证分辨率正确）
                imgs = []
                for p in batch_paths:
                    t = load_image_tensor(p, self.device, size=img_size)
                    if t is not None:
                        imgs.append(t)
                if not imgs:
                    continue
                orig_batch = torch.cat(imgs, dim=0)  # [B, 3, H, W]

                # 计算 oracle 特征（动态生成，20-50 步 PGD）
                oracle_features = compute_oracle_features_pgd(
                    orig_batch, encoders,
                    pgd_steps=self.stage_cfg.get("oracle_pgd_steps", 20),
                    pgd_alpha=self.stage_cfg.get("oracle_pgd_alpha", 1/255),
                    pgd_eps=self.stage_cfg.get("oracle_pgd_eps", 16/255),
                )

                # 前向传播
                optimizer.zero_grad()
                adv_batch = model(orig_batch)

                # 计算损失
                total_loss, loss_dict = loss_fn(
                    adv_batch, orig_batch, encoders, oracle_features, augmentor
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 低通平滑（去除高频伪影）
                with torch.no_grad():
                    adv_smooth = model.apply_lowpass_smoothing(
                        adv_batch.detach(), orig_batch
                    )

                epoch_losses.append(loss_dict)
                epoch_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            scheduler.step()

            # Epoch 统计
            avg_total = sum(d["total"] for d in epoch_losses) / max(len(epoch_losses), 1)
            avg_distort = sum(d["distort"] for d in epoch_losses) / max(len(epoch_losses), 1)

            # PSNR 估计（从最后一个 batch）
            psnr = compute_psnr(orig_batch[:1], adv_batch[:1].detach())

            log_msg = (f"Epoch {epoch:3d}/{self.stage_cfg['epochs']} | "
                       f"Loss={avg_total:.4f} | Distort={avg_distort:.5f} | PSNR={psnr:.1f}dB")
            self.logger.info(log_msg)

            history.append({"epoch": epoch, "loss": avg_total,
                             "distort": avg_distort, "psnr": psnr})

            # 保存 checkpoint
            if epoch % self.stage_cfg.get("save_interval", 5) == 0 or epoch == self.stage_cfg["epochs"]:
                ckpt_path = os.path.join(self.run_dir, "checkpoints",
                                         f"epoch_{epoch:04d}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_total,
                    "config": cfg.STEGO_MODEL_CONFIG,
                }, ckpt_path)

                if avg_total < best_loss:
                    best_loss = avg_total
                    best_path = os.path.join(self.run_dir, "checkpoints", "best.pt")
                    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                                "loss": avg_total, "config": cfg.STEGO_MODEL_CONFIG},
                               best_path)
                    self.logger.info(f"  → 新最优 checkpoint 保存: {best_path}")

        # 保存历史
        hist_path = os.path.join(self.run_dir, "history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        self.logger.info(f"训练完成！最优 Loss: {best_loss:.4f}")
        self.logger.info(f"历史记录: {hist_path}")
        return model, history
