"""
Test 3: 验证 StegoEncoder 的前向传播、DCT 约束、低通平滑
运行：conda run -n deeplearning python test/3_test_stego_encoder.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import config
from models.stego_encoder import (
    StegoEncoder, dct2d_blockwise, idct2d_blockwise,
    make_mid_freq_mask, gaussian_lowpass_2d
)


def test_dct_roundtrip():
    """DCT → iDCT 应为近似恒等变换"""
    x = torch.rand(2, 3, 64, 64)
    dct = dct2d_blockwise(x, patch_size=8)
    recon = idct2d_blockwise(dct, patch_size=8)
    err = (x - recon).abs().max().item()
    assert err < 1e-4, f"DCT 往返误差过大: {err:.6f}"
    print(f"✓ DCT 往返误差 {err:.2e} < 1e-4")


def test_mid_freq_mask():
    """中频 mask 只有中间频率为 1"""
    mask = make_mid_freq_mask(8, band_low=3, band_high=15,
                              device=torch.device("cpu"))
    assert mask.shape == (1, 1, 8, 8)
    total_ones = mask.sum().item()
    assert total_ones == 13, f"band 3-15 共 13 个系数，实际={total_ones}"
    # DC 项（[0,0]）和高频应为 0
    assert mask[0, 0, 0, 0].item() == 0.0, "DC 项应被遮蔽"
    print(f"✓ 中频 mask 正确：{int(total_ones)} 个非零系数")


def test_lowpass_smoothing():
    """低通平滑应降低高频能量"""
    noisy = torch.randn(1, 3, 64, 64) * 0.1
    smooth = gaussian_lowpass_2d(noisy, sigma=1.0)
    assert smooth.shape == noisy.shape
    # 平滑后 L2 范数应 ≤ 原始（高频被抑制）
    assert smooth.norm() <= noisy.norm() + 1e-6
    print(f"✓ 低通平滑有效: {noisy.norm():.3f} → {smooth.norm():.3f}")


def test_stego_forward_shape():
    """前向传播：输入输出尺寸一致"""
    model = StegoEncoder(config.STEGO_MODEL_CONFIG)
    model.eval()
    for H, W in [(64, 64), (128, 96), (224, 224)]:
        x = torch.rand(1, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, f"输出尺寸不匹配 {out.shape} vs {x.shape}"
        print(f"  ✓ 尺寸 ({H},{W}) → 输出 {tuple(out.shape)}")
    print("✓ 前向传播尺寸正确")


def test_stego_value_range():
    """输出值域必须在 [0, 1]"""
    model = StegoEncoder(config.STEGO_MODEL_CONFIG)
    model.eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.min() >= -1e-6, f"输出最小值异常: {out.min():.4f}"
    assert out.max() <= 1 + 1e-6, f"输出最大值异常: {out.max():.4f}"
    print(f"✓ 值域 [{out.min():.4f}, {out.max():.4f}] ⊆ [0, 1]")


def test_stego_epsilon_constraint():
    """扰动 L∞ 范数必须 ≤ epsilon"""
    cfg = config.STEGO_MODEL_CONFIG.copy()
    cfg["epsilon"] = 16 / 255
    model = StegoEncoder(cfg)
    model.eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    delta = (out - x).abs().max().item()
    eps = cfg["epsilon"] + 1e-5
    assert delta <= eps, f"扰动 L∞={delta:.4f} 超过 epsilon={cfg['epsilon']:.4f}"
    print(f"✓ 扰动 L∞={delta:.4f} ≤ ε={cfg['epsilon']:.4f}")


def test_stego_gradient_flow():
    """梯度必须能反传到模型参数"""
    model = StegoEncoder(config.STEGO_MODEL_CONFIG)
    model.train()
    x = torch.rand(1, 3, 64, 64, requires_grad=False)
    out = model(x)
    loss = (out - x).pow(2).mean()
    loss.backward()
    # 检查至少有一个参数有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "没有参数收到梯度！"
    print("✓ 梯度正常反传")


def test_stego_lowpass_smoothing():
    """apply_lowpass_smoothing 应减小高频扰动"""
    model = StegoEncoder(config.STEGO_MODEL_CONFIG)
    model.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        adv = model(x)
        smoothed = model.apply_lowpass_smoothing(adv, x)
    assert smoothed.shape == x.shape
    delta_before = (adv - x).abs().max().item()
    delta_after  = (smoothed - x).abs().max().item()
    assert delta_after <= delta_before + 1e-5
    print(f"✓ 低通平滑后扰动 L∞: {delta_before:.4f} → {delta_after:.4f}")


def test_parameter_count():
    """打印参数量，确保在轻量级范围内"""
    model = StegoEncoder(config.STEGO_MODEL_CONFIG)
    total  = model.parameter_count()
    train  = model.trainable_parameter_count()
    print(f"✓ 参数量: 总计={total/1e6:.2f}M, 可训练={train/1e6:.2f}M")
    assert total < 100e6, f"模型过大: {total/1e6:.1f}M 参数"


if __name__ == "__main__":
    print("=" * 55)
    print("Test 3: StegoEncoder 功能验证")
    print("=" * 55)
    test_dct_roundtrip()
    test_mid_freq_mask()
    test_lowpass_smoothing()
    test_stego_forward_shape()
    test_stego_value_range()
    test_stego_epsilon_constraint()
    test_stego_gradient_flow()
    test_stego_lowpass_smoothing()
    test_parameter_count()
    print("\n✅ 所有 StegoEncoder 测试通过")
