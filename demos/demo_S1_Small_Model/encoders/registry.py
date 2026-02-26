"""
编码器注册表 — 支持零侵入式扩展
使用：在新编码器文件中加 @register_encoder("name") 即可
"""
from typing import Dict, Type, List, TYPE_CHECKING

if TYPE_CHECKING:
    from encoders.base import BaseVisualEncoder

ENCODER_REGISTRY: Dict[str, Type["BaseVisualEncoder"]] = {}


def register_encoder(name: str):
    """装饰器：将编码器类注册到全局注册表"""
    def decorator(cls):
        if name in ENCODER_REGISTRY:
            raise ValueError(f"编码器 '{name}' 已注册，请使用不同的名称")
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def list_encoders() -> List[str]:
    """返回所有已注册编码器的名称列表"""
    return list(ENCODER_REGISTRY.keys())


def get_encoder_class(name: str) -> Type["BaseVisualEncoder"]:
    """按名称获取编码器类（未实例化）"""
    if name not in ENCODER_REGISTRY:
        available = list(ENCODER_REGISTRY.keys())
        raise KeyError(f"未找到编码器 '{name}'。已注册: {available}")
    return ENCODER_REGISTRY[name]
