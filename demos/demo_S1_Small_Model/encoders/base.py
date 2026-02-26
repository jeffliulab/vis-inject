"""
视觉编码器基类 — 所有编码器必须实现此接口
新增编码器：继承 BaseVisualEncoder，加 @register_encoder("name") 即可
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from torch import Tensor


class BaseVisualEncoder(ABC):
    """
    视觉编码器的统一抽象接口。
    每个子类对应一个 MLLM 的视觉编码器部分（不含 LLM 头）。
    """

    def __init__(self, cfg: dict):
        """
        cfg: ENCODER_CONFIG 中对应模型的字典，包含 model_id, img_size, weight 等
        """
        self.cfg = cfg
        self.weight: float = cfg.get("weight", 1.0)
        self._model = None   # 延迟加载

    # ---- 必须实现的属性 ----

    @property
    @abstractmethod
    def name(self) -> str:
        """注册名称，与 @register_encoder 中的字符串一致"""
        ...

    @property
    def img_size(self) -> int:
        """期望输入分辨率（正方形边长，单位像素）"""
        return self.cfg["img_size"]

    @property
    def norm_mean(self) -> List[float]:
        """模型专用归一化均值，[R, G, B] 顺序，值域 [0,1]"""
        return self.cfg["norm_mean"]

    @property
    def norm_std(self) -> List[float]:
        """模型专用归一化标准差，[R, G, B] 顺序"""
        return self.cfg["norm_std"]

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """编码器输出的特征向量维度"""
        ...

    @property
    def device(self) -> str:
        return self.cfg.get("device", "cuda")

    # ---- 必须实现的方法 ----

    @abstractmethod
    def load(self) -> None:
        """加载模型权重到 self._model（仅编码器部分，冻结参数）"""
        ...

    @abstractmethod
    def encode(self, images: Tensor) -> Tensor:
        """
        提取视觉特征。
        输入：[B, 3, img_size, img_size]，已按 norm_mean/std 归一化，值域约 [-2, 2]
        输出：[B, D] 或 [B, N_tokens, D]，将在 losses.py 中做 mean pooling
        """
        ...

    # ---- 可选覆盖：注意力引导 ----

    def get_attention_map(self, images: Tensor) -> Optional[Tensor]:
        """
        返回注意力图用于空间权重 mask（可选实现）。
        输出：[B, H, W] 或 None（不支持时返回 None）
        """
        return None

    # ---- 工具方法（子类通常不需要覆盖）----

    def is_loaded(self) -> bool:
        return self._model is not None

    def normalize(self, images: Tensor) -> Tensor:
        """将 [0,1] 图像张量按本编码器的 mean/std 归一化"""
        mean = torch.tensor(self.norm_mean, device=images.device,
                            dtype=images.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self.norm_std,  device=images.device,
                            dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    def resize_and_encode(self, images: Tensor) -> Tensor:
        """
        完整流程：resize → normalize → encode
        输入：[B, 3, H, W]，值域 [0,1]，H/W 不限
        输出：[B, D]
        """
        import torch.nn.functional as F
        if images.shape[-1] != self.img_size or images.shape[-2] != self.img_size:
            images = F.interpolate(images, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
        normed = self.normalize(images)
        feats  = self.encode(normed)
        # 统一返回 [B, D]（token 维度做 mean pooling）
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        return feats

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}(name={self.name}, img_size={self.img_size}, {loaded})"
