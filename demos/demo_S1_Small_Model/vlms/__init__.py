"""vlms 包初始化"""
from vlms.registry import VLM_REGISTRY, register_vlm, list_vlms, get_vlm_class
from vlms.base import BaseVLM

from vlms import blip2_vlm      # noqa: F401
from vlms import deepseek_vlm   # noqa: F401
from vlms import qwen_vlm       # noqa: F401


def load_vlms(names: list, vlm_config: dict) -> list:
    """按名称列表实例化 VLM（不自动加载权重，调用 vlm.load() 按需加载）"""
    result = []
    for name in names:
        cls = get_vlm_class(name)
        cfg = vlm_config.get(name, {})
        result.append(cls(cfg))
    return result


__all__ = ["VLM_REGISTRY", "register_vlm", "list_vlms", "get_vlm_class", "load_vlms", "BaseVLM"]
