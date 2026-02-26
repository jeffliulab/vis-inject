"""VLM 注册表"""
from typing import Dict, Type, List, TYPE_CHECKING

if TYPE_CHECKING:
    from vlms.base import BaseVLM

VLM_REGISTRY: Dict[str, Type["BaseVLM"]] = {}


def register_vlm(name: str):
    def decorator(cls):
        if name in VLM_REGISTRY:
            raise ValueError(f"VLM '{name}' 已注册")
        VLM_REGISTRY[name] = cls
        return cls
    return decorator


def list_vlms() -> List[str]:
    return list(VLM_REGISTRY.keys())


def get_vlm_class(name: str) -> Type["BaseVLM"]:
    if name not in VLM_REGISTRY:
        raise KeyError(f"未找到 VLM '{name}'。已注册: {list(VLM_REGISTRY.keys())}")
    return VLM_REGISTRY[name]
