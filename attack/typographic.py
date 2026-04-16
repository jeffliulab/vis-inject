"""
VisInject C2 — Typographic Injection Attack
=============================================
Renders target text directly onto the image at varying visibility levels.
Exploits VLMs' text bias: they preferentially process visible text in images.

Three sub-variants:
  - high_visible:   White text with dark outline, clearly readable (FigStep-style)
  - low_visible:    Background-adaptive color, semi-transparent (harder to spot)
  - scene_blend:    Text styled as watermark/annotation fitting the image context

Reference: FigStep (AAAI'25 Oral, 82.5% ASR), SceneTAP (2025)

Usage:
    from attack.typographic import TypographicAttack
    atk = TypographicAttack("visit www.example.com", variant="low_visible")
    adv_image = atk.attack(clean_image)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from attack.base import AttackBase


def _get_dominant_color(image: Image.Image, region: tuple) -> tuple:
    """Get the dominant color in a region for adaptive text coloring."""
    crop = image.crop(region)
    arr = np.array(crop)
    avg = arr.mean(axis=(0, 1)).astype(int)
    return tuple(avg)


def _contrast_color(bg_color: tuple) -> tuple:
    """Return a color that's slightly different from bg but not too visible."""
    # Shift each channel by 30-50 towards opposite
    shift = 40
    return tuple(
        min(255, c + shift) if c < 128 else max(0, c - shift)
        for c in bg_color[:3]
    )


class TypographicAttack(AttackBase):
    """Render target text onto image at varying visibility levels."""

    name = "typographic"
    category = "C2"

    VARIANTS = ("high_visible", "low_visible", "scene_blend")

    def __init__(self, target_phrase: str, variant: str = "low_visible",
                 alpha: float = None, font_size: int = None,
                 position: str = "bottom", color: tuple = None):
        """
        Args:
            target_phrase: Text to inject into the image.
            variant: One of 'high_visible', 'low_visible', 'scene_blend'.
            alpha: Text opacity (0.0-1.0). Auto-set by variant if None.
            font_size: Font size in pixels. Auto-scaled if None.
            position: Text position ('top', 'bottom', 'center', 'random').
            color: RGB text color. Auto-set by variant if None.
        """
        # Set defaults per variant
        defaults = {
            "high_visible": {"alpha": 0.9, "font_size_ratio": 0.06},
            "low_visible": {"alpha": 0.25, "font_size_ratio": 0.04},
            "scene_blend": {"alpha": 0.15, "font_size_ratio": 0.035},
        }
        v = defaults.get(variant, defaults["low_visible"])

        self.variant = variant
        self.alpha = alpha if alpha is not None else v["alpha"]
        self.font_size_ratio = v["font_size_ratio"]
        self.font_size_override = font_size
        self.position = position
        self.color_override = color

        super().__init__(target_phrase, variant=variant, alpha=self.alpha,
                         position=position)

    def attack(self, clean_image: Image.Image) -> Image.Image:
        w, h = clean_image.size
        font_size = self.font_size_override or max(12, int(h * self.font_size_ratio))

        # Try to use a good font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except (IOError, OSError):
                font = ImageFont.load_default()

        # Create text layer
        text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # Measure text
        bbox = draw.textbbox((0, 0), self.target_phrase, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Position
        if self.position == "bottom":
            x, y = (w - tw) // 2, h - th - int(h * 0.05)
        elif self.position == "top":
            x, y = (w - tw) // 2, int(h * 0.05)
        elif self.position == "center":
            x, y = (w - tw) // 2, (h - th) // 2
        else:  # random
            import random
            x = random.randint(0, max(0, w - tw))
            y = random.randint(0, max(0, h - th))

        # Color selection
        if self.color_override:
            text_color = self.color_override
        elif self.variant == "high_visible":
            text_color = (255, 255, 255)
        elif self.variant == "scene_blend":
            # Adaptive: pick color close to background
            region = (max(0, x), max(0, y), min(w, x + tw), min(h, y + th))
            bg = _get_dominant_color(clean_image, region)
            text_color = _contrast_color(bg)
        else:  # low_visible
            text_color = (200, 200, 200)

        alpha_int = int(255 * self.alpha)

        # Draw text
        if self.variant == "high_visible":
            # Draw dark outline for readability
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), self.target_phrase,
                              font=font, fill=(0, 0, 0, alpha_int))

        draw.text((x, y), self.target_phrase, font=font,
                  fill=(*text_color, alpha_int))

        # Composite
        result = clean_image.convert("RGBA")
        result = Image.alpha_composite(result, text_layer)
        return result.convert("RGB")


# Convenience constructors
def high_visible(target_phrase: str, **kwargs) -> TypographicAttack:
    return TypographicAttack(target_phrase, variant="high_visible", **kwargs)

def low_visible(target_phrase: str, **kwargs) -> TypographicAttack:
    return TypographicAttack(target_phrase, variant="low_visible", **kwargs)

def scene_blend(target_phrase: str, **kwargs) -> TypographicAttack:
    return TypographicAttack(target_phrase, variant="scene_blend", **kwargs)
