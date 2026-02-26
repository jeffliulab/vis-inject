"""
注入目标基类 — 定义"注入什么"和"如何判断成功"
新增：继承 BasePromptTarget，加 @register_prompt("name") 即可
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor


class BasePromptTarget(ABC):
    """
    注入目标的统一抽象接口。
    将"注入内容"和"成功检测"封装在同一个类中，
    使 rewards.py 和 evaluate.py 与具体 prompt 类型解耦。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ---- 必须实现 ----

    @property
    @abstractmethod
    def name(self) -> str:
        """注册名称"""
        ...

    @property
    @abstractmethod
    def target_text(self) -> str:
        """
        注入目标的文字描述。
        在模式 A（固定 Token）中，这是训练时的参考文字；
        在模式 B（可控 Prompt）中，作为 FiLM 条件输入的候选之一。
        """
        ...

    @abstractmethod
    def compute_success(self, model_response: str) -> float:
        """
        检测 VLM 的回复中注入是否成功。
        返回值：[0.0, 1.0]，1.0 为完全成功，0.0 为完全失败。
        用于 RL 奖励计算和评估指标。
        """
        ...

    # ---- 可选覆盖 ----

    def get_text_embedding(self, text_encoder=None) -> Optional[Tensor]:
        """
        获取目标文字的嵌入向量（用于 BLIP-2 ITC 损失）。
        text_encoder: CLIP text encoder 或 BLIP-2 text encoder。
        默认返回 None（不参与 ITC 损失）。
        """
        return None

    def get_description(self) -> str:
        """人类可读的注入目标描述（用于日志）"""
        return f"{self.name}: '{self.target_text}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, target='{self.target_text[:40]}...')"
