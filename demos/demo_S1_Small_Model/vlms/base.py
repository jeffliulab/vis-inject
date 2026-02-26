"""
VLM 基类 — 统一多模态模型的推理接口
新增 VLM：继承 BaseVLM，加 @register_vlm("name") 即可
"""
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    from encoders.base import BaseVisualEncoder


class BaseVLM(ABC):
    """
    完整多模态大模型的统一抽象接口（用于 Stage 2 RL 和评估）。
    只需实现 generate() 方法即可接入训练/评估流程。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self._processor = None

    # ---- 必须实现 ----

    @property
    @abstractmethod
    def name(self) -> str:
        """注册名称"""
        ...

    @abstractmethod
    def load(self) -> None:
        """加载模型和 processor（整个 VLM，包含 LLM 部分）"""
        ...

    @abstractmethod
    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = 100) -> str:
        """
        统一生成接口：给定 PIL 图像和问题文本，返回模型的文字回复。
        各子类负责处理自身的输入格式差异。
        """
        ...

    # ---- 可选覆盖 ----

    def get_visual_encoder(self) -> Optional["BaseVisualEncoder"]:
        """
        返回该 VLM 对应的 BaseVisualEncoder 实例（可选实现）。
        用于 Stage 1C 端到端训练时共享编码器，避免重复加载。
        """
        return None

    # ---- 工具方法 ----

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self.cfg.get("device", "cuda")

    @property
    def max_new_tokens(self) -> int:
        return self.cfg.get("max_new_tokens", 100)

    def __repr__(self) -> str:
        loaded = "loaded" if self.is_loaded() else "not loaded"
        return f"{self.__class__.__name__}(name={self.name}, {loaded})"
