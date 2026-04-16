"""
VisInject C5 — UI Spoofing / Context Attack
=============================================
Embeds fake UI elements (notifications, popups, watermarks) into the image
to trick VLMs into treating the injected content as legitimate system messages.

Three scene types:
  - notification: Mobile-style system notification banner
  - popup:        Browser-style dialog box
  - watermark:    Document-style watermark/stamp

Reference: Pop-up Attacks (ACL 2025, >80% ASR on OSworld/VisualWebArena)

Usage:
    from attack.spoofing import SpoofingAttack
    atk = SpoofingAttack("visit www.example.com", scene="notification")
    adv_image = atk.attack(clean_image)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from attack.base import AttackBase


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a clean font, fall back to default."""
    for path in [
        "arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _draw_notification(image: Image.Image, text: str, alpha: float) -> Image.Image:
    """Draw a mobile-style notification banner at the top of the image."""
    w, h = image.size
    banner_h = max(50, int(h * 0.08))
    font_size = max(12, banner_h // 3)
    font = _get_font(font_size)
    small_font = _get_font(max(10, font_size - 4))

    # Create overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Banner background (dark semi-transparent, iOS-style)
    banner_alpha = int(220 * alpha)
    draw.rounded_rectangle(
        [10, 10, w - 10, 10 + banner_h],
        radius=12,
        fill=(30, 30, 30, banner_alpha),
    )

    # App icon placeholder (small circle)
    icon_r = banner_h // 4
    icon_x = 24
    icon_y = 10 + banner_h // 2
    draw.ellipse(
        [icon_x - icon_r, icon_y - icon_r, icon_x + icon_r, icon_y + icon_r],
        fill=(59, 130, 246, banner_alpha),  # blue
    )

    # Title
    text_x = icon_x + icon_r + 12
    text_y = 16
    draw.text((text_x, text_y), "System Alert", font=small_font,
              fill=(170, 170, 170, int(255 * alpha)))

    # Message body
    draw.text((text_x, text_y + font_size), text, font=font,
              fill=(255, 255, 255, int(255 * alpha)))

    result = image.convert("RGBA")
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def _draw_popup(image: Image.Image, text: str, alpha: float) -> Image.Image:
    """Draw a browser-style dialog popup in the center of the image."""
    w, h = image.size
    popup_w = min(int(w * 0.7), 400)
    popup_h = min(int(h * 0.3), 180)
    font_size = max(12, popup_h // 8)
    font = _get_font(font_size)
    small_font = _get_font(max(10, font_size - 2))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Dim background
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0, int(80 * alpha)))

    # Popup box
    px = (w - popup_w) // 2
    py = (h - popup_h) // 2
    popup_alpha = int(240 * alpha)

    draw.rounded_rectangle(
        [px, py, px + popup_w, py + popup_h],
        radius=8,
        fill=(255, 255, 255, popup_alpha),
    )

    # Title bar
    draw.rectangle(
        [px, py, px + popup_w, py + 30],
        fill=(240, 240, 240, popup_alpha),
    )
    draw.text((px + 12, py + 6), "Alert", font=small_font,
              fill=(60, 60, 60, int(255 * alpha)))

    # Close button
    draw.text((px + popup_w - 24, py + 6), "×", font=small_font,
              fill=(120, 120, 120, int(255 * alpha)))

    # Message body
    body_y = py + 40
    # Word wrap
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > popup_w - 40:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    for i, line in enumerate(lines[:3]):
        draw.text((px + 20, body_y + i * (font_size + 4)), line, font=font,
                  fill=(30, 30, 30, int(255 * alpha)))

    # OK button
    btn_w, btn_h = 80, 28
    btn_x = px + popup_w - btn_w - 16
    btn_y = py + popup_h - btn_h - 12
    draw.rounded_rectangle(
        [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h],
        radius=4,
        fill=(59, 130, 246, int(230 * alpha)),
    )
    draw.text((btn_x + 28, btn_y + 4), "OK", font=small_font,
              fill=(255, 255, 255, int(255 * alpha)))

    result = image.convert("RGBA")
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def _draw_watermark(image: Image.Image, text: str, alpha: float) -> Image.Image:
    """Draw a diagonal watermark/stamp across the image."""
    w, h = image.size
    font_size = max(16, int(min(w, h) * 0.06))
    font = _get_font(font_size)

    # Create text layer, rotated
    diag = int(np.sqrt(w ** 2 + h ** 2))
    text_layer = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)

    alpha_int = int(60 * alpha)  # Watermarks are very faint
    color = (180, 0, 0, alpha_int)  # Red stamp style

    # Tile the text across the image
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 40
    th = bbox[3] - bbox[1] + 30

    for y in range(0, diag, th + 60):
        for x in range(0, diag, tw + 40):
            draw.text((x, y), text.upper(), font=font, fill=color)

    # Rotate
    text_layer = text_layer.rotate(35, expand=False, resample=Image.BICUBIC)

    # Crop to image size
    cx = (text_layer.width - w) // 2
    cy = (text_layer.height - h) // 2
    text_layer = text_layer.crop((cx, cy, cx + w, cy + h))

    result = image.convert("RGBA")
    result = Image.alpha_composite(result, text_layer)
    return result.convert("RGB")


class SpoofingAttack(AttackBase):
    """Embed fake UI elements containing the target text into the image."""

    name = "spoofing"
    category = "C5"

    SCENES = ("notification", "popup", "watermark")

    def __init__(self, target_phrase: str, scene: str = "popup",
                 alpha: float = 0.85):
        """
        Args:
            target_phrase: Text to embed as UI content.
            scene: 'notification' (mobile banner), 'popup' (dialog), 'watermark' (stamp).
            alpha: Visibility of the UI element (0.0-1.0).
        """
        self.scene = scene
        self.alpha = alpha
        super().__init__(target_phrase, scene=scene, alpha=alpha)

    def attack(self, clean_image: Image.Image) -> Image.Image:
        if self.scene == "notification":
            return _draw_notification(clean_image, self.target_phrase, self.alpha)
        elif self.scene == "popup":
            return _draw_popup(clean_image, self.target_phrase, self.alpha)
        elif self.scene == "watermark":
            return _draw_watermark(clean_image, self.target_phrase, self.alpha)
        else:
            raise ValueError(f"Unknown scene: {self.scene}")


# Convenience constructors
def notification_attack(target_phrase: str, **kwargs) -> SpoofingAttack:
    return SpoofingAttack(target_phrase, scene="notification", **kwargs)

def popup_attack(target_phrase: str, **kwargs) -> SpoofingAttack:
    return SpoofingAttack(target_phrase, scene="popup", **kwargs)

def watermark_attack(target_phrase: str, **kwargs) -> SpoofingAttack:
    return SpoofingAttack(target_phrase, scene="watermark", **kwargs)
