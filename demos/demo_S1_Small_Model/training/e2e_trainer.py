"""
Stage 1A 端到端训练（Full-Model Gradient）

核心思路：
  类似 demo3 的 PGD 攻击，但用 StegoEncoder 替代 PGD delta。
  通过完整 Qwen2.5-VL-3B 的 CE Loss 反向传播来训练 StegoEncoder。

  StegoEncoder(orig) → adv_img → Qwen(pixel_values) → CE Loss on target_text → ∇ StegoEncoder

内存分析（4090 16GB）：
  Qwen 3B 参数 (bf16)：~6GB
  StegoEncoder + 激活 (f32)：~1GB
  Qwen 前向激活 (batch=1, seq≈400)：~3-4GB
  总计：~10-11GB（可行）

速度估算：
  每张图：~3-5s（Qwen 前向+反向）
  200 图 × 20 epoch = 4000 步 × 4s ≈ 4.5 小时
  加上 checkpoint 保存：~5 小时

训练监控：
  - CE loss 应从 ~7-10 开始，逐步下降
  - PSNR 应始终 > 20 dB（StegoEncoder 有 epsilon 约束）
  - 每 5 epoch 保存 checkpoint
"""

import os
import sys
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# 关键优化：设置 CUDA 内存分配器使用可扩展段，避免内存碎片
# 没有此设置，16GB GPU 上连续训练会因碎片导致 100× 速度下降
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
# TF32 加速（Ampere 架构及以上，4090 支持）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from models.stego_encoder import StegoEncoder


# ============================================================
# 工具函数
# ============================================================

def setup_logging(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, "train.log")
    logger = logging.getLogger("e2e_trainer")
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
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    paths = [str(p) for p in data_dir.iterdir()
             if p.suffix.lower() in extensions]
    random.shuffle(paths)
    return paths[:num_images]


def load_image_tensor(path: str, size: int = 392) -> Optional[torch.Tensor]:
    """加载图像为 [1, 3, size, size] float32 [0,1]"""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = size / min(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        left = (nw - size) // 2
        top  = (nh - size) // 2
        img  = img.crop((left, top, left + size, top + size))
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    except Exception as e:
        return None


def compute_psnr(orig: torch.Tensor, adv: torch.Tensor) -> float:
    mse = F.mse_loss(orig.float().cpu(), adv.float().cpu()).item()
    if mse < 1e-10:
        return 100.0
    import math
    return 10 * math.log10(1.0 / mse)


# ============================================================
# Qwen 全模型加载器（仅用于计算 CE loss，参数冻结）
# ============================================================

class QwenAttackLoss(nn.Module):
    """
    使用完整 Qwen2.5-VL-3B 计算对抗 CE loss。
    参数全部冻结，只用于 loss 计算。
    移植自 demo3/model_loader.py，适配 StegoEncoder 输出。
    """

    # Qwen 视觉归一化参数
    IMG_MEAN = [0.48145466, 0.4578275,  0.40821073]
    IMG_STD  = [0.26862954, 0.26130258, 0.27577711]

    # 视觉编码器配置
    PATCH_SIZE = 14
    MERGE_SIZE = 2
    TEMPORAL_PATCH_SIZE = 2

    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.dtype = torch.bfloat16

        print(f"[QwenAttackLoss] 加载完整模型: {model_id}")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype, device_map="auto"
        )
        self.model.eval()
        self.model.requires_grad_(False)
        # 梯度检查点：用重新计算替代存储中间激活，节省 ~50% 显存
        # 这使得 4090 能够稳定运行 backward pass（避免 GPU-CPU swap 导致的极慢速度）
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass  # 部分版本不支持，忽略

        # 图像归一化参数
        self.img_mean = torch.tensor(
            self.IMG_MEAN, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.img_std = torch.tensor(
            self.IMG_STD, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        # 计算 grid_thw（固定尺寸 392×392）
        H = W = 392  # StegoEncoder 输出尺寸
        self.grid_thw = torch.tensor(
            [[1, H // self.PATCH_SIZE, W // self.PATCH_SIZE]],
            dtype=torch.long, device=device
        )

        print(f"[QwenAttackLoss] 加载完成")

    def _image_to_pixel_values(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        [1, 3, H, W] float32 [0,1] → pixel_values [N_patches, C*T*pH*pW] bfloat16
        与 demo3 完全一致的 patch 提取方式。
        """
        x = (image_tensor.to(self.device, dtype=torch.float32) - self.img_mean) / self.img_std
        x = x.to(self.dtype)

        B, C, H, W = x.shape
        PS = self.PATCH_SIZE
        MS = self.MERGE_SIZE
        T  = self.TEMPORAL_PATCH_SIZE

        h_merge = H // (PS * MS)
        w_merge = W // (PS * MS)

        x = x.reshape(B, C, h_merge, MS, PS, w_merge, MS, PS)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, T, -1, -1)
        pixel_values = x.reshape(-1, C * T * PS * PS)
        return pixel_values

    def _build_input_ids_and_labels(self, target_text: str, question: str):
        """
        构建 input_ids 和 labels（只在 target_text 部分有有效 label）。
        完全移植自 demo3 的 _build_attack_ids()。
        """
        n_image_tokens = (392 // self.PATCH_SIZE // self.MERGE_SIZE) ** 2  # 196

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": (
                f"<|vision_start|>{'<|image_pad|>' * n_image_tokens}<|vision_end|>{question}"
            )},
            {"role": "assistant", "content": target_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        target_ids = self.tokenizer.encode(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        target_len = target_ids.shape[1]

        im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        seq = input_ids[0].tolist()
        im_end_pos = len(seq) - 1 - seq[::-1].index(im_end_id)
        target_start = im_end_pos - target_len

        labels = torch.full_like(input_ids, -100)
        labels[0, target_start:im_end_pos] = input_ids[0, target_start:im_end_pos]
        return input_ids, labels

    def forward(self, adv_img: torch.Tensor,
                target_text: str,
                question: str = "Describe this image.") -> torch.Tensor:
        """
        计算 StegoEncoder 输出图像对应的 CE loss（使 Qwen 倾向于生成 target_text）。

        adv_img: [1, 3, 392, 392] float32 [0,1]（StegoEncoder 输出，需要梯度）
        返回: scalar CE loss（梯度连接到 adv_img → StegoEncoder）
        """
        pixel_values = self._image_to_pixel_values(adv_img)
        input_ids, labels = self._build_input_ids_and_labels(target_text, question)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=self.grid_thw,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, img_tensor: torch.Tensor,
                 question: str = "Describe this image.",
                 max_new_tokens: int = 100) -> str:
        """
        从图像张量生成文本回答（推理用，不带梯度）。
        """
        try:
            pixel_values = self._image_to_pixel_values(img_tensor)
            n_image_tokens = (392 // self.PATCH_SIZE // self.MERGE_SIZE) ** 2

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": (
                    f"<|vision_start|>{'<|image_pad|>' * n_image_tokens}<|vision_end|>{question}"
                )},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=self.grid_thw,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            trimmed = generated_ids[0, input_ids.shape[1]:]
            return self.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        except Exception as e:
            return f"[ERROR: {e}]"


# ============================================================
# 端到端训练器
# ============================================================

class E2ETrainer:
    """
    端到端训练：通过完整 Qwen VLM 的 CE loss 训练 StegoEncoder。
    配置来自 config.STAGE1A_CONFIG。
    """

    def __init__(self, stage: str = "1a"):
        self.stage = stage
        self.stage_cfg = cfg.STAGE1A_CONFIG if stage == "1a" else cfg.STAGE1B_CONFIG
        self.device = cfg.DEVICE

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("logs_and_outputs", f"stage{stage}_e2e_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        self.logger = setup_logging(self.run_dir)
        self.logger.info(f"E2E Stage {stage.upper()} 训练初始化")
        self.logger.info(f"输出目录: {self.run_dir}")

    def run(self):
        """执行完整的端到端训练"""
        self.logger.info("=" * 60)
        self.logger.info("开始 E2E Stage 1A 训练（StegoEncoder + Full Qwen CE Loss）")
        self.logger.info(f"  图像数量: {self.stage_cfg['num_images']}")
        self.logger.info(f"  Epochs:   {self.stage_cfg['epochs']}")
        self.logger.info(f"  LR:       {self.stage_cfg['lr']}")
        self.logger.info(f"  Epsilon:  {cfg.STEGO_MODEL_CONFIG['epsilon']:.4f}")
        self.logger.info("=" * 60)

        torch.manual_seed(cfg.SEED)

        # ---- 目标文本 ----
        from prompts import load_prompt
        prompt_target = load_prompt(cfg.ACTIVE_PROMPT, cfg.PROMPT_CONFIG)
        target_text = prompt_target.target_text
        question    = cfg.EVAL_CONFIG.get("question", "Describe this image.")
        self.logger.info(f"目标文本: '{target_text}'")
        self.logger.info(f"问题: '{question}'")

        # ---- 加载 Qwen 完整模型 ----
        model_id = cfg.ENCODER_CONFIG["qwen"]["model_id"]
        qwen_loss = QwenAttackLoss(model_id, device=self.device)

        # ---- StegoEncoder ----
        stego = StegoEncoder(cfg.STEGO_MODEL_CONFIG).to(self.device)
        self.logger.info(f"StegoEncoder 参数量: {stego.parameter_count()/1e6:.2f}M")

        optimizer = optim.AdamW(stego.parameters(),
                                lr=self.stage_cfg["lr"],
                                weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.stage_cfg["epochs"], eta_min=1e-6
        )

        # ---- 数据 ----
        img_size = 392  # 必须是 28 的倍数（Qwen patch_size=14 × merge_size=2）
        image_paths = load_image_paths("data/train", self.stage_cfg["num_images"])
        self.logger.info(f"图像数量: {len(image_paths)}")

        # ---- 训练循环 ----
        history = []
        best_loss = float("inf")

        for epoch in range(1, self.stage_cfg["epochs"] + 1):
            stego.train()
            epoch_losses = []
            epoch_psnrs  = []
            random.shuffle(image_paths)

            epoch_bar = tqdm(image_paths, desc=f"Epoch {epoch}/{self.stage_cfg['epochs']}",
                             leave=False)

            for img_path in epoch_bar:
                orig_t = load_image_tensor(img_path, size=img_size)
                if orig_t is None:
                    continue
                orig_t = orig_t.to(self.device)

                # ---- 前向：StegoEncoder → adv_img ----
                optimizer.zero_grad()
                adv_t = stego(orig_t.float())  # [1, 3, 392, 392]

                # ---- CE Loss 通过完整 Qwen ----
                ce_loss = qwen_loss(adv_t, target_text, question)

                # ---- 额外：L2 失真惩罚（保持 PSNR）----
                l2_loss = F.mse_loss(adv_t, orig_t.detach())
                total_loss = ce_loss + cfg.LOSS_WEIGHTS.get("distort", 0.5) * l2_loss

                # ---- 反向传播 ----
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(stego.parameters(), max_norm=1.0)
                optimizer.step()

                # ---- 低通平滑 ----
                with torch.no_grad():
                    adv_smooth = stego.apply_lowpass_smoothing(adv_t.detach(), orig_t)

                # ---- 指标 ----
                psnr_val = compute_psnr(orig_t, adv_t.detach())
                epoch_losses.append({"ce": ce_loss.item(), "total": total_loss.item()})
                epoch_psnrs.append(psnr_val)

                epoch_bar.set_postfix({
                    "ce": f"{ce_loss.item():.3f}",
                    "psnr": f"{psnr_val:.1f}"
                })

            scheduler.step()

            # ---- Epoch 统计 ----
            avg_ce   = sum(d["ce"]    for d in epoch_losses) / max(len(epoch_losses), 1)
            avg_loss = sum(d["total"] for d in epoch_losses) / max(len(epoch_losses), 1)
            avg_psnr = sum(epoch_psnrs) / max(len(epoch_psnrs), 1)

            log_msg = (f"Epoch {epoch:3d}/{self.stage_cfg['epochs']} | "
                       f"CE={avg_ce:.4f} | Total={avg_loss:.4f} | PSNR={avg_psnr:.1f}dB")
            self.logger.info(log_msg)

            history.append({"epoch": epoch, "ce_loss": avg_ce, "total_loss": avg_loss,
                            "psnr": avg_psnr})

            # ---- 快速 ASR 检查（每 5 epoch）----
            if epoch % 5 == 0 and image_paths:
                sample_path = image_paths[0]
                sample_t = load_image_tensor(sample_path, size=img_size)
                if sample_t is not None:
                    with torch.no_grad():
                        stego.eval()
                        adv_sample = stego(sample_t.to(self.device).float())
                        response = qwen_loss.generate(adv_sample, question=question)
                        stego.train()
                    triggered = prompt_target.target_text.lower() in response.lower()
                    self.logger.info(f"  [Quick ASR] 触发: {'✓' if triggered else '✗'} | "
                                    f"回答: '{response[:80]}'")

            # ---- 保存 checkpoint ----
            if epoch % self.stage_cfg.get("save_interval", 5) == 0 or epoch == self.stage_cfg["epochs"]:
                ckpt_path = os.path.join(self.run_dir, "checkpoints",
                                         f"epoch_{epoch:04d}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": stego.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_ce,
                    "config": cfg.STEGO_MODEL_CONFIG,
                }, ckpt_path)

                if avg_ce < best_loss:
                    best_loss = avg_ce
                    best_path = os.path.join(self.run_dir, "checkpoints", "best.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state": stego.state_dict(),
                        "loss": avg_ce,
                        "config": cfg.STEGO_MODEL_CONFIG,
                    }, best_path)
                    self.logger.info(f"  → 新最优 checkpoint: {best_path} (CE={avg_ce:.4f})")

        # ---- 保存历史 ----
        hist_path = os.path.join(self.run_dir, "history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        self.logger.info(f"训练完成！最优 CE Loss: {best_loss:.4f}")
        return stego, history
