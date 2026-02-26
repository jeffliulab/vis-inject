"""
RL 奖励函数

自动遍历已注册的 VLM 列表和 PromptTarget 接口计算奖励信号。
新增 VLM 或 Prompt 类型，此文件无需修改。

总奖励 R = λ_trigger · TriggerSuccess
          - λ_distort · DistortionPenalty
          + λ_robust  · RobustBonus
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image
import numpy as np


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """[3, H, W] 或 [1, 3, H, W] 张量 → PIL Image（uint8）"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    arr = (img_tensor.detach().cpu().clamp(0, 1).numpy()
           .transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_distortion_penalty(adv_img: torch.Tensor,
                               orig_img: torch.Tensor) -> float:
    """L2 失真惩罚（标量）"""
    return F.mse_loss(adv_img.detach(), orig_img.detach()).item()


def compute_trigger_success(vlm, adv_pil: Image.Image,
                            question: str,
                            prompt_target) -> float:
    """
    单个 VLM 上的触发成功率。
    调用 VLM.generate() 并用 PromptTarget.compute_success() 评分。
    """
    try:
        response = vlm.generate(adv_pil, question)
        return prompt_target.compute_success(response)
    except Exception as e:
        print(f"  [WARNING] {vlm.name} generate 失败: {e}")
        return 0.0


class RewardComputer:
    """
    RL 奖励计算器。
    在 Stage 2 REINFORCE 训练中，对每个 episode（图像）计算标量奖励。
    """

    def __init__(self, weights: dict, question: str = "请描述这张图片中的内容。"):
        """
        weights: config.REWARD_WEIGHTS，含 trigger/distort/robust 权重
        question: 发给 VLM 的固定提问
        """
        self.w = weights
        self.question = question

    def compute(
        self,
        adv_img: torch.Tensor,
        orig_img: torch.Tensor,
        vlms: list,
        prompt_target,
        augmentor=None,
        vlm_index: int = 0,
    ) -> dict:
        """
        计算单张对抗图像的奖励（轮流在不同 VLM 上评估）。

        参数：
            adv_img:      [1, 3, H, W] 对抗图像
            orig_img:     [1, 3, H, W] 原始图像
            vlms:         List[BaseVLM] 已加载的 VLM 列表
            prompt_target: BasePromptTarget 当前注入目标
            augmentor:    DifferentiableAugmentor，用于鲁棒性测试
            vlm_index:    当前 episode 使用哪个 VLM（轮流策略）

        返回：
            reward_dict = {"total": float, "trigger": float, "distort": float, "robust": float}
        """
        if not vlms:
            return {"total": 0.0, "trigger": 0.0, "distort": 0.0, "robust": 0.0}

        # 选择当前 episode 的 VLM（轮流）
        vlm = vlms[vlm_index % len(vlms)]

        # ---- 1. 触发成功奖励 ----
        adv_pil = tensor_to_pil(adv_img)
        trigger_score = compute_trigger_success(vlm, adv_pil, self.question, prompt_target)

        # ---- 2. 鲁棒性奖励（增强后仍然触发）----
        robust_score = 0.0
        if augmentor is not None:
            try:
                aug_img = augmentor.robustness_aug(adv_img.detach(), distortion="random")
                aug_pil = tensor_to_pil(aug_img)
                robust_score = compute_trigger_success(
                    vlm, aug_pil, self.question, prompt_target
                )
            except Exception as e:
                print(f"  [WARNING] 鲁棒性测试失败: {e}")

        # ---- 3. 失真惩罚 ----
        distort_penalty = compute_distortion_penalty(adv_img, orig_img)

        # ---- 加权总奖励 ----
        total = (self.w.get("trigger", 1.0) * trigger_score
               - self.w.get("distort", 0.3) * distort_penalty
               + self.w.get("robust",  0.5) * robust_score)

        return {
            "total":   total,
            "trigger": trigger_score,
            "distort": distort_penalty,
            "robust":  robust_score,
            "vlm":     vlm.name,
        }

    def compute_batch(
        self,
        adv_imgs: torch.Tensor,
        orig_imgs: torch.Tensor,
        vlms: list,
        prompt_target,
        augmentor=None,
        start_vlm_index: int = 0,
    ) -> List[dict]:
        """
        对一个 batch 的图像逐张计算奖励。
        返回 List[reward_dict]，长度 = batch_size。
        """
        rewards = []
        for i in range(adv_imgs.shape[0]):
            r = self.compute(
                adv_img=adv_imgs[i:i+1],
                orig_img=orig_imgs[i:i+1],
                vlms=vlms,
                prompt_target=prompt_target,
                augmentor=augmentor,
                vlm_index=start_vlm_index + i,
            )
            rewards.append(r)
        return rewards

    @staticmethod
    def rewards_to_tensor(reward_dicts: list,
                          device: torch.device) -> torch.Tensor:
        """将 reward_dict 列表转为标量张量 [B]，用于 REINFORCE 更新"""
        totals = [r["total"] for r in reward_dicts]
        return torch.tensor(totals, dtype=torch.float32, device=device)
