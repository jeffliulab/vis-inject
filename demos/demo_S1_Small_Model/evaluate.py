"""
评估脚本 — 自动遍历 VLM × 失真条件矩阵

输出：
  - ASR（Attack Success Rate）矩阵
  - PSNR / SSIM / LPIPS 图像质量指标
  - 结构化 JSON + 可视化对比图
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict
import random

import torch
from PIL import Image
from tqdm import tqdm

import config as cfg
from models.stego_encoder import StegoEncoder
from augmentation import DifferentiableAugmentor
from utils import (
    load_image, pil_to_tensor, tensor_to_pil,
    calculate_psnr, calculate_ssim, visualize_comparison, setup_run_dir
)


def setup_logging(run_dir: str) -> logging.Logger:
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(run_dir, "eval.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


class Evaluator:
    """
    多模型 × 多失真条件评估器。
    自动遍历 VLM 注册表和 DISTORTION_SUITE，无需手动为每个模型写分支。
    """

    def __init__(self, checkpoint_path: str, num_test_images: int = None):
        self.device = cfg.DEVICE
        self.checkpoint_path = checkpoint_path
        self.num_images = num_test_images or cfg.EVAL_CONFIG.get("num_test_images", 50)
        self.distortions = cfg.EVAL_CONFIG.get("distortion_suite", ["none", "jpeg_q50"])
        self.question = cfg.EVAL_CONFIG.get("question", "请描述这张图片。")

        self.run_dir = setup_run_dir("logs_and_outputs", prefix="eval")
        self.logger  = setup_logging(self.run_dir)

    def _load_model(self) -> StegoEncoder:
        model = StegoEncoder(cfg.STEGO_MODEL_CONFIG).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.logger.info(f"加载模型: {self.checkpoint_path}")
        return model

    def _load_vlms(self) -> list:
        from vlms import load_vlms
        vlms = load_vlms(cfg.ACTIVE_VLMS, cfg.VLM_CONFIG)
        for vlm in vlms:
            self.logger.info(f"加载 VLM: {vlm.name} ...")
            vlm.load()
        return vlms

    def _load_prompt_target(self):
        from prompts import load_prompt
        return load_prompt(cfg.ACTIVE_PROMPT, cfg.PROMPT_CONFIG)

    def _load_test_images(self) -> List[str]:
        """加载测试图像路径（从 data/test 或 data/train 子集）"""
        from pathlib import Path
        for d in ["data/test", "data/train", "sample"]:
            p = Path(d)
            if p.exists():
                paths = [str(x) for x in p.iterdir()
                         if x.suffix.lower() in (".jpg", ".jpeg", ".png")]
                if paths:
                    random.shuffle(paths)
                    return paths[:self.num_images]
        # 无测试图像时使用合成图
        self.logger.warning("未找到测试图像，使用合成随机图像")
        return [f"__synthetic_{i}__" for i in range(self.num_images)]

    def run(self) -> dict:
        self.logger.info("=" * 60)
        self.logger.info("开始评估")
        self.logger.info(f"  VLM: {cfg.ACTIVE_VLMS}")
        self.logger.info(f"  Prompt: {cfg.ACTIVE_PROMPT}")
        self.logger.info(f"  失真条件: {self.distortions}")
        self.logger.info(f"  测试图像数: {self.num_images}")
        self.logger.info("=" * 60)

        model = self._load_model()
        vlms  = self._load_vlms()
        prompt_target = self._load_prompt_target()
        augmentor = DifferentiableAugmentor()
        test_paths = self._load_test_images()

        # ---- 结果矩阵 ----
        # results[vlm_name][distortion] = {asr, psnr, ssim}
        results: Dict[str, Dict[str, dict]] = {
            vlm.name: {d: {"asr": [], "psnr": [], "ssim": []} for d in self.distortions}
            for vlm in vlms
        }

        vis_dir = os.path.join(self.run_dir, "visualizations")

        for img_idx, path in enumerate(tqdm(test_paths, desc="评估图像")):
            # 加载原始图像
            if path.startswith("__synthetic_"):
                orig_t = torch.rand(1, 3, 256, 256, device=self.device)
                orig_pil = tensor_to_pil(orig_t)
            else:
                try:
                    orig_pil = load_image(path, size=(256, 256))
                    orig_t   = pil_to_tensor(orig_pil, self.device)
                except Exception:
                    continue

            # 生成对抗图像
            with torch.no_grad():
                adv_t = model(orig_t)
            adv_pil = tensor_to_pil(adv_t)

            # PSNR / SSIM
            psnr = calculate_psnr(orig_t.cpu(), adv_t.cpu())
            ssim = calculate_ssim(orig_t.cpu(), adv_t.cpu())

            # 保存可视化（前 5 张）
            if img_idx < 5:
                vis_path = os.path.join(vis_dir, f"img_{img_idx:03d}.png")
                visualize_comparison(orig_pil, adv_pil, vis_path,
                                     title=f"Image {img_idx} | PSNR={psnr:.1f}dB")

            # 逐 VLM 逐失真评估
            for vlm in vlms:
                for dist in self.distortions:
                    try:
                        if dist == "none":
                            test_pil = adv_pil
                        else:
                            aug_t = augmentor.robustness_aug(
                                adv_t.clone(), distortion=dist
                            )
                            test_pil = tensor_to_pil(aug_t)

                        response = vlm.generate(test_pil, self.question)
                        score = prompt_target.compute_success(response)
                    except Exception as e:
                        self.logger.warning(f"  {vlm.name}/{dist} 失败: {e}")
                        score = 0.0

                    results[vlm.name][dist]["asr"].append(score)
                    results[vlm.name][dist]["psnr"].append(psnr)
                    results[vlm.name][dist]["ssim"].append(ssim)

        # ---- 汇总结果 ----
        summary = {}
        for vlm_name, dist_dict in results.items():
            summary[vlm_name] = {}
            for dist, metrics in dist_dict.items():
                n = max(len(metrics["asr"]), 1)
                summary[vlm_name][dist] = {
                    "asr":  sum(metrics["asr"])  / n,
                    "psnr": sum(metrics["psnr"]) / n,
                    "ssim": sum(metrics["ssim"]) / n,
                    "n":    n,
                }

        # ---- 打印结果表格 ----
        self.logger.info("\n=== 评估结果 ===")
        header = f"{'':20s}" + "".join(f"{d:>14s}" for d in self.distortions)
        self.logger.info(header)
        for vlm_name in summary:
            row = f"{vlm_name:20s}"
            for dist in self.distortions:
                asr = summary[vlm_name][dist]["asr"] * 100
                row += f"{asr:13.1f}%"
            self.logger.info(row)

        self.logger.info("\nPSNR / SSIM (averaged over all VLMs):")
        for dist in self.distortions:
            all_psnr = [summary[v][dist]["psnr"] for v in summary]
            all_ssim = [summary[v][dist]["ssim"] for v in summary]
            self.logger.info(f"  {dist:>15s}: PSNR={sum(all_psnr)/len(all_psnr):.1f}dB, "
                             f"SSIM={sum(all_ssim)/len(all_ssim):.3f}")

        # ---- 保存 JSON ----
        result_path = os.path.join(self.run_dir, "results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "config": {
                "checkpoint": self.checkpoint_path,
                "active_vlms": cfg.ACTIVE_VLMS,
                "active_prompt": cfg.ACTIVE_PROMPT,
                "num_images": self.num_images,
            }}, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n结果保存至: {result_path}")
        return summary
