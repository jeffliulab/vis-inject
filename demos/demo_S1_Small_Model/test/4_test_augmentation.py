"""
Test 4: 验证 augmentation.py 的各种增强操作
运行：conda run -n deeplearning python test/4_test_augmentation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from augmentation import (
    input_diversity, quantize_ste, jpeg_like_compression,
    gaussian_blur, screenshot_simulate, scale_and_restore,
    DifferentiableAugmentor,
)


def make_input(B=2, C=3, H=64, W=64):
    return torch.rand(B, C, H, W)


def test_input_diversity_shape():
    """输出应为 [B, 3, target_size, target_size]"""
    x = make_input()
    for tgt in [64, 128, 224, 392]:
        out = input_diversity(x, target_size=tgt, prob=1.0)
        assert out.shape == (2, 3, tgt, tgt), f"shape mismatch: {out.shape}"
    print("✓ input_diversity 尺寸正确")


def test_input_diversity_value_range():
    x = make_input()
    out = input_diversity(x, 64, prob=1.0)
    assert out.min() >= -1e-4 and out.max() <= 1 + 1e-4
    print(f"✓ input_diversity 值域 [{out.min():.3f}, {out.max():.3f}]")


def test_input_diversity_gradient():
    x = make_input(B=1).requires_grad_(True)
    out = input_diversity(x, 64, prob=1.0)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("✓ input_diversity 梯度可传播")


def test_quantize_ste_shape():
    x = make_input()
    out = quantize_ste(x)
    assert out.shape == x.shape
    # 检查梯度（STE 应允许梯度通过）
    x2 = make_input(B=1).requires_grad_(True)
    quantize_ste(x2).sum().backward()
    assert x2.grad is not None
    print("✓ quantize_ste 形状和梯度正确")


def test_jpeg_compression_shape():
    x = make_input()
    for q in [50, 30, 80]:
        out = jpeg_like_compression(x, quality=q)
        assert out.shape == x.shape, f"JPEG Q{q} shape mismatch"
        assert out.min() >= -1e-4 and out.max() <= 1 + 1e-4
    print("✓ jpeg_like_compression 形状和值域正确")


def test_jpeg_compression_gradient():
    x = make_input(B=1).requires_grad_(True)
    out = jpeg_like_compression(x, quality=50)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("✓ jpeg_like_compression 梯度可传播")


def test_gaussian_blur_shape():
    x = make_input()
    out = gaussian_blur(x, sigma=1.0)
    assert out.shape == x.shape
    # 模糊后与原图不同
    assert (out - x).abs().mean() > 1e-5
    print(f"✓ gaussian_blur 形状正确，平均变化={( out - x).abs().mean():.4f}")


def test_scale_and_restore():
    x = make_input()
    for s in [0.5, 0.75, 1.5]:
        out = scale_and_restore(x, scale=s)
        assert out.shape == x.shape
    print("✓ scale_and_restore 还原尺寸正确")


def test_screenshot_simulate():
    x = make_input()
    out = screenshot_simulate(x)
    assert out.shape == x.shape
    assert out.min() >= -1e-4 and out.max() <= 1 + 1e-4
    print("✓ screenshot_simulate 正常")


def test_augmentor_all_distortions():
    aug = DifferentiableAugmentor()
    x = make_input()
    distortions = [
        "none", "jpeg_q50", "jpeg_q30", "scale_half",
        "scale_double", "gaussian_blur", "screenshot_sim",
    ]
    for d in distortions:
        out = aug.robustness_aug(x, distortion=d)
        assert out.shape == x.shape, f"{d}: shape mismatch"
        assert out.min() >= -1e-3 and out.max() <= 1 + 1e-3, \
            f"{d}: 值域异常 [{out.min():.3f}, {out.max():.3f}]"
        print(f"  ✓ {d}: OK")
    print("✓ 所有失真模式正常")


def test_augmentor_diversity():
    aug = DifferentiableAugmentor()
    x = make_input(B=1)
    for tgt in [224, 392]:
        out = aug.diversity_aug(x, tgt, prob=1.0)
        assert out.shape == (1, 3, tgt, tgt)
    print("✓ diversity_aug 尺寸正确")


if __name__ == "__main__":
    print("=" * 55)
    print("Test 4: augmentation.py 验证")
    print("=" * 55)
    test_input_diversity_shape()
    test_input_diversity_value_range()
    test_input_diversity_gradient()
    test_quantize_ste_shape()
    test_jpeg_compression_shape()
    test_jpeg_compression_gradient()
    test_gaussian_blur_shape()
    test_scale_and_restore()
    test_screenshot_simulate()
    test_augmentor_all_distortions()
    test_augmentor_diversity()
    print("\n✅ 所有 augmentation 测试通过")
