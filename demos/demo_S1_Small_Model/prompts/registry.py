"""Prompt 注册表"""
from typing import Dict, Type, List, TYPE_CHECKING

if TYPE_CHECKING:
    from prompts.base import BasePromptTarget

PROMPT_REGISTRY: Dict[str, Type["BasePromptTarget"]] = {}


def register_prompt(name: str):
    def decorator(cls):
        if name in PROMPT_REGISTRY:
            raise ValueError(f"Prompt '{name}' 已注册")
        PROMPT_REGISTRY[name] = cls
        return cls
    return decorator


def list_prompts() -> List[str]:
    return list(PROMPT_REGISTRY.keys())


def get_prompt_class(name: str) -> Type["BasePromptTarget"]:
    if name not in PROMPT_REGISTRY:
        raise KeyError(f"未找到 Prompt '{name}'。已注册: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]
