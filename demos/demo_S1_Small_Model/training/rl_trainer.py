"""
Stage 2 REINFORCE 微调循环

将 StegoEncoder 视为策略网络，VLM 作为黑盒环境。
单步 MDP：每个 episode = 一张图像 → 生成对抗图 → 查询 VLM → 获得奖励
算法：REINFORCE with baseline（running mean baseline 减少方差）
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import random
from typing import List

import torch
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from models.stego_encoder import StegoEncoder
from rewards import RewardComputer
from augmentation import DifferentiableAugmentor
from training.proxy_trainer import setup_logging, load_image_paths, load_image_tensor, compute_psnr


class RLTrainer:
    """
    Stage 2 REINFORCE 微调。
    从 Stage 1 checkpoint 加载 StegoEncoder，在真实 VLM 上优化触发率。
    VLM 作为黑盒 oracle，无需对 LLM 反向传播。
    """

    def __init__(self, checkpoint_path: str):
        self.device = cfg.DEVICE
        self.checkpoint_path = checkpoint_path

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("logs_and_outputs", f"stage2_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        self.logger = setup_logging(self.run_dir)
        self.logger.info(f"Stage 2 RL 训练初始化，checkpoint: {checkpoint_path}")

    def _load_vlms(self) -> list:
        from vlms import load_vlms
        self.logger.info(f"加载 VLM: {cfg.ACTIVE_VLMS}")
        vlms = load_vlms(cfg.ACTIVE_VLMS, cfg.VLM_CONFIG)
        for vlm in vlms:
            self.logger.info(f"  加载 {vlm.name} ...")
            vlm.load()
            self.logger.info(f"  {vlm.name} 加载完成")
        return vlms

    def _load_prompt_target(self):
        from prompts import load_prompt
        return load_prompt(cfg.ACTIVE_PROMPT, cfg.PROMPT_CONFIG)

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("开始 Stage 2 REINFORCE 微调")
        self.logger.info(f"  VLM: {cfg.ACTIVE_VLMS}")
        self.logger.info(f"  Prompt: {cfg.ACTIVE_PROMPT}")
        self.logger.info("=" * 60)

        torch.manual_seed(cfg.SEED)

        # 加载预训练 StegoEncoder
        model = StegoEncoder(cfg.STEGO_MODEL_CONFIG).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        self.logger.info(f"加载 checkpoint: epoch={ckpt.get('epoch', '?')}, "
                         f"loss={ckpt.get('loss', '?'):.4f}")

        vlms = self._load_vlms()
        prompt_target = self._load_prompt_target()
        augmentor = DifferentiableAugmentor()
        reward_computer = RewardComputer(
            cfg.REWARD_WEIGHTS,
            question=cfg.EVAL_CONFIG["question"]
        )

        optimizer = optim.Adam(model.parameters(), lr=cfg.STAGE2_CONFIG["lr"])
        image_paths = load_image_paths("data/train", 1000)  # RL 用较少图像

        # ---- REINFORCE 参数 ----
        episodes_per_update = cfg.STAGE2_CONFIG.get("episodes_per_update", 16)
        max_updates = cfg.STAGE2_CONFIG.get("max_updates", 2000)
        baseline = 0.0          # 运行均值 baseline（减少方差）
        baseline_alpha = 0.05   # baseline EMA 系数

        history = []
        best_asr = 0.0
        vlm_index = 0

        pbar = tqdm(range(max_updates), desc="RL Updates")

        for update_idx in pbar:
            model.train()
            log_probs_list = []
            rewards_list   = []

            # 收集 episodes_per_update 个 episode
            batch_paths = random.choices(image_paths, k=episodes_per_update)

            for path in batch_paths:
                orig = load_image_tensor(path, self.device)
                if orig is None:
                    continue

                # 策略：生成对抗图像
                adv = model(orig)

                # log_prob 近似：对输出像素值的对数似然（连续动作空间简化）
                delta = adv - orig
                # 使用像素扰动的 L2 范数作为代理 log_prob
                log_prob = -0.5 * delta.pow(2).mean()
                log_probs_list.append(log_prob)

                # 奖励（黑盒 VLM 查询，不可微）
                reward_dict = reward_computer.compute(
                    adv_img=adv.detach(),
                    orig_img=orig,
                    vlms=vlms,
                    prompt_target=prompt_target,
                    augmentor=augmentor,
                    vlm_index=vlm_index,
                )
                rewards_list.append(reward_dict["total"])
                vlm_index += 1

            if not log_probs_list:
                continue

            # REINFORCE 更新：∇J ≈ Σ (R - b) · ∇log_π
            rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32,
                                          device=self.device)
            advantages = rewards_tensor - baseline

            # 更新 baseline
            baseline = (baseline_alpha * rewards_tensor.mean().item()
                        + (1 - baseline_alpha) * baseline)

            log_probs_tensor = torch.stack(log_probs_list[:len(rewards_list)])
            policy_loss = -(advantages.detach() * log_probs_tensor).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # 统计
            mean_reward  = rewards_tensor.mean().item()
            mean_trigger = sum(r["trigger"] for r in
                               reward_computer.compute_batch(
                                   torch.cat([load_image_tensor(p, self.device)
                                              for p in batch_paths[:1]], 0),
                                   torch.cat([load_image_tensor(p, self.device)
                                              for p in batch_paths[:1]], 0),
                                   vlms=[], prompt_target=prompt_target
                               )) if False else 0.0  # 简化，不重复查询

            log_interval = cfg.STAGE2_CONFIG.get("log_interval", 20)
            if update_idx % log_interval == 0:
                self.logger.info(
                    f"Update {update_idx:4d}/{max_updates} | "
                    f"Policy Loss={policy_loss.item():.4f} | "
                    f"Mean Reward={mean_reward:.4f} | "
                    f"Baseline={baseline:.4f}"
                )
                history.append({
                    "update": update_idx,
                    "policy_loss": policy_loss.item(),
                    "mean_reward": mean_reward,
                    "baseline": baseline,
                })

            pbar.set_postfix({"reward": f"{mean_reward:.3f}",
                              "loss": f"{policy_loss.item():.4f}"})

            # 保存 checkpoint
            save_interval = cfg.STAGE2_CONFIG.get("save_interval", 100)
            if (update_idx + 1) % save_interval == 0:
                ckpt_path = os.path.join(
                    self.run_dir, "checkpoints", f"update_{update_idx+1:05d}.pt"
                )
                torch.save({
                    "update": update_idx,
                    "model_state": model.state_dict(),
                    "mean_reward": mean_reward,
                    "config": cfg.STEGO_MODEL_CONFIG,
                }, ckpt_path)

        # 保存最终模型
        final_path = os.path.join(self.run_dir, "checkpoints", "final.pt")
        torch.save({"model_state": model.state_dict(),
                    "config": cfg.STEGO_MODEL_CONFIG}, final_path)

        hist_path = os.path.join(self.run_dir, "history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        self.logger.info(f"RL 训练完成！最终模型: {final_path}")
        return model, history
