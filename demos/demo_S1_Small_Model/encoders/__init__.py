"""
encoders 包初始化 — 自动导入所有已实现的编码器，触发注册
"""
from encoders.registry import ENCODER_REGISTRY, register_encoder, list_encoders, get_encoder_class
from encoders.base import BaseVisualEncoder

# 自动导入所有编码器实现（触发 @register_encoder 装饰器）
from encoders import blip2_encoder   # noqa: F401
from encoders import deepseek_encoder  # noqa: F401
from encoders import qwen_encoder    # noqa: F401


def load_encoders(names: list, encoder_config: dict) -> list:
    """
    按名称列表实例化编码器。
    返回 List[BaseVisualEncoder]，按传入顺序排列。
    """
    encoders = []
    for name in names:
        cls = get_encoder_class(name)
        cfg = encoder_config.get(name, {})
        enc = cls(cfg)
        encoders.append(enc)
    return encoders


__all__ = [
    "ENCODER_REGISTRY",
    "register_encoder",
    "list_encoders",
    "get_encoder_class",
    "load_encoders",
    "BaseVisualEncoder",
]
