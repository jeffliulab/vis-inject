"""
VisInject Attack Base Class
============================
Unified interface for all v2.0 attack methods.

All attacks take a clean image + target phrase and produce an adversarial image.
"""

import os
from abc import ABC, abstractmethod

from PIL import Image


class AttackBase(ABC):
    """Base class for all VisInject attack methods."""

    name: str = "base"
    category: str = "unknown"  # C1-C5

    def __init__(self, target_phrase: str, **kwargs):
        self.target_phrase = target_phrase
        self.params = kwargs

    @abstractmethod
    def attack(self, clean_image: Image.Image) -> Image.Image:
        """Apply attack to a clean image, return adversarial image."""
        ...

    def attack_file(self, clean_path: str, output_path: str) -> dict:
        """Attack a file and save result. Returns metadata dict."""
        clean = Image.open(clean_path).convert("RGB")
        adv = self.attack(clean)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        adv.save(output_path)

        psnr = self._compute_psnr(clean, adv)
        return {
            "attack": self.name,
            "category": self.category,
            "target_phrase": self.target_phrase,
            "clean_image": os.path.basename(clean_path),
            "adv_image": os.path.basename(output_path),
            "psnr": round(psnr, 2),
            "params": self.params,
        }

    @staticmethod
    def _compute_psnr(img_a: Image.Image, img_b: Image.Image) -> float:
        """Compute PSNR between two PIL images."""
        import numpy as np
        a = np.array(img_a).astype(float)
        b = np.array(img_b).astype(float)
        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(255.0 ** 2 / mse)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, target={self.target_phrase!r})"
