"""
VisInject C3 — Steganographic Injection Attack
================================================
Embeds target text invisibly into image pixels using steganographic techniques.
The text is invisible to humans but may be detected by VLMs during processing.

Two sub-methods:
  - lsb:  Adaptive Least Significant Bit embedding (simple, fast)
  - dct:  DCT frequency-domain embedding (more robust to JPEG compression)

Reference: "Invisible Injections" (2025, 31.8% ASR with neural stego)

Usage:
    from attack.steganography import SteganographicAttack
    atk = SteganographicAttack("visit www.example.com", method="lsb")
    adv_image = atk.attack(clean_image)
"""

import hashlib

import numpy as np
from PIL import Image

from attack.base import AttackBase


def _text_to_bits(text: str) -> list[int]:
    """Convert text string to list of bits (UTF-8 encoding)."""
    data = text.encode("utf-8")
    # Prepend length as 32-bit integer for extraction
    length = len(data)
    header = length.to_bytes(4, "big")
    payload = header + data
    bits = []
    for byte in payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _generate_positions(seed: str, total_pixels: int, num_positions: int) -> list[int]:
    """Generate pseudorandom pixel positions from a seed (cryptographic selection)."""
    h = hashlib.sha256(seed.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    return rng.choice(total_pixels, size=num_positions, replace=False).tolist()


def _texture_variance(arr: np.ndarray, x: int, y: int, w: int, h: int,
                      window: int = 3) -> float:
    """Compute local texture variance around a pixel."""
    x0 = max(0, x - window)
    y0 = max(0, y - window)
    x1 = min(w, x + window + 1)
    y1 = min(h, y + window + 1)
    patch = arr[y0:y1, x0:x1].astype(float)
    return patch.var()


class SteganographicAttack(AttackBase):
    """Embed target text invisibly in image pixels."""

    name = "steganography"
    category = "C3"

    METHODS = ("lsb", "dct")

    def __init__(self, target_phrase: str, method: str = "lsb",
                 bit_depth: int = 1, channel: int = 2,
                 adaptive: bool = True, seed: str = "visinject",
                 dct_strength: float = 5.0):
        """
        Args:
            target_phrase: Text to embed in the image.
            method: 'lsb' or 'dct'.
            bit_depth: Number of LSBs to modify (1-3). Higher = more capacity but more visible.
            channel: Color channel to embed in (0=R, 1=G, 2=B). Blue is least perceptible.
            adaptive: If True, prefer embedding in high-texture regions.
            seed: Seed for pseudorandom position generation.
            dct_strength: Embedding strength for DCT method.
        """
        self.method = method
        self.bit_depth = bit_depth
        self.channel = channel
        self.adaptive = adaptive
        self.seed = seed
        self.dct_strength = dct_strength

        super().__init__(target_phrase, method=method, bit_depth=bit_depth,
                         channel=channel, adaptive=adaptive)

    def attack(self, clean_image: Image.Image) -> Image.Image:
        if self.method == "lsb":
            return self._lsb_embed(clean_image)
        elif self.method == "dct":
            return self._dct_embed(clean_image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _lsb_embed(self, image: Image.Image) -> Image.Image:
        """Adaptive LSB steganography."""
        arr = np.array(image).copy()
        h, w, c = arr.shape
        bits = _text_to_bits(self.target_phrase)

        if len(bits) > h * w:
            raise ValueError(f"Image too small ({h*w} pixels) for message ({len(bits)} bits)")

        # Generate embedding positions
        positions = _generate_positions(self.seed, h * w, len(bits))

        if self.adaptive:
            # Sort positions by texture variance (prefer complex regions)
            variances = []
            for pos in positions:
                py, px = divmod(pos, w)
                var = _texture_variance(arr, px, py, w, h)
                variances.append(var)
            # Re-sort: highest variance first
            sorted_indices = np.argsort(variances)[::-1]
            positions = [positions[i] for i in sorted_indices[:len(bits)]]

        # Embed bits
        ch = self.channel
        mask = ~((1 << self.bit_depth) - 1) & 0xFF  # Clear target bits

        for bit, pos in zip(bits, positions):
            py, px = divmod(pos, w)
            val = int(arr[py, px, ch])
            val = (val & mask) | (bit << (self.bit_depth - 1))
            arr[py, px, ch] = val

        return Image.fromarray(arr)

    def _dct_embed(self, image: Image.Image) -> Image.Image:
        """DCT frequency-domain steganography."""
        from scipy.fft import dctn, idctn

        arr = np.array(image).astype(float)
        h, w, c = arr.shape
        bits = _text_to_bits(self.target_phrase)
        ch = self.channel

        # Work on 8×8 blocks (standard JPEG block size)
        block_size = 8
        blocks_h = h // block_size
        blocks_w = w // block_size

        if len(bits) > blocks_h * blocks_w:
            raise ValueError(f"Image too small for message in DCT mode")

        # Select blocks pseudorandomly
        total_blocks = blocks_h * blocks_w
        block_positions = _generate_positions(self.seed, total_blocks, len(bits))

        channel_data = arr[:, :, ch].copy()

        for bit, bpos in zip(bits, block_positions):
            by, bx = divmod(bpos, blocks_w)
            y0 = by * block_size
            x0 = bx * block_size
            block = channel_data[y0:y0 + block_size, x0:x0 + block_size]

            # Forward DCT
            dct_block = dctn(block, type=2, norm="ortho")

            # Embed in mid-frequency coefficient (4,3) — survives JPEG compression
            target_coeff = dct_block[4, 3]
            if bit == 1:
                dct_block[4, 3] = abs(target_coeff) + self.dct_strength
            else:
                dct_block[4, 3] = -(abs(target_coeff) + self.dct_strength)

            # Inverse DCT
            channel_data[y0:y0 + block_size, x0:x0 + block_size] = idctn(
                dct_block, type=2, norm="ortho"
            )

        arr[:, :, ch] = np.clip(channel_data, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


# Convenience constructors
def lsb_attack(target_phrase: str, **kwargs) -> SteganographicAttack:
    return SteganographicAttack(target_phrase, method="lsb", **kwargs)

def dct_attack(target_phrase: str, **kwargs) -> SteganographicAttack:
    return SteganographicAttack(target_phrase, method="dct", **kwargs)
