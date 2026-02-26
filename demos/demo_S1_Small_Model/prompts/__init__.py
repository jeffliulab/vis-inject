"""prompts 包初始化"""
from prompts.registry import PROMPT_REGISTRY, register_prompt, list_prompts, get_prompt_class
from prompts.base import BasePromptTarget

from prompts import fixed_keyword         # noqa: F401
from prompts import style_injection       # noqa: F401
from prompts import instruction_injection # noqa: F401


def load_prompt(name: str, prompt_config: dict) -> BasePromptTarget:
    """实例化指定名称的 PromptTarget"""
    cls = get_prompt_class(name)
    cfg = prompt_config.get(name, {})
    return cls(cfg)


__all__ = [
    "PROMPT_REGISTRY", "register_prompt", "list_prompts",
    "get_prompt_class", "load_prompt", "BasePromptTarget",
]
