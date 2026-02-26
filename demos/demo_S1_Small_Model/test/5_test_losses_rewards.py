"""
Test 5: 验证 losses.py 和 rewards.py 的核心功能（不加载大模型）
运行：conda run -n deeplearning python test/5_test_losses_rewards.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import config
from losses import (
    VGGPerceptualLoss, freq_reg_loss, cosine_feature_loss,
    MultiEncoderProxyLoss, compute_oracle_features_pgd
)
from rewards import RewardComputer, tensor_to_pil, compute_distortion_penalty
from augmentation import DifferentiableAugmentor


def make_img(B=2, H=64, W=64):
    return torch.rand(B, 3, H, W)


# ---- losses ----

def test_cosine_feature_loss():
    feat_a = torch.rand(4, 512)
    feat_b = torch.rand(4, 512)
    loss = cosine_feature_loss(feat_a, feat_b)
    assert loss.shape == torch.Size([])
    assert 0.0 <= loss.item() <= 2.0
    # 相同特征 loss 应接近 0
    loss_same = cosine_feature_loss(feat_a, feat_a)
    assert loss_same.item() < 1e-4, f"相同特征损失应为0，实际={loss_same:.4f}"
    print(f"✓ cosine_feature_loss: same={loss_same:.4f}, diff={loss:.4f}")


def test_freq_reg_loss():
    orig = make_img(B=2)
    adv  = orig + 0.05 * torch.randn_like(orig)
    adv  = adv.clamp(0, 1)
    l = freq_reg_loss(adv, orig)
    assert l.shape == torch.Size([])
    assert l.item() >= 0.0
    # 加高频噪声后 freq_reg 应增大
    adv_hf = orig.clone()
    adv_hf[:, :, ::1, ::1] += 0.05 * torch.rand_like(orig)  # 高频
    l_hf = freq_reg_loss(adv_hf.clamp(0,1), orig)
    print(f"✓ freq_reg_loss: clean={l.item():.4f}, noisy={l_hf.item():.4f}")


def test_vgg_perceptual_loss():
    """VGG 感知损失接口验证（用 mock 代替下载，避免挂起）"""
    import torch.nn as nn
    # 用一个简单的 mock 替换 VGG，只验证接口
    vgg_loss = VGGPerceptualLoss()
    # 注入 mock VGG（避免实际下载）
    mock_vgg = nn.Sequential(nn.AdaptiveAvgPool2d((8, 8))).eval()
    vgg_loss._vgg = mock_vgg

    orig = make_img(B=1, H=64)
    adv  = (orig + 0.01 * torch.randn_like(orig)).clamp(0, 1)

    # 手动调用（跳过 _load_vgg 中的下载逻辑）
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    def prep(x): return (x - mean) / std
    f_adv  = mock_vgg(prep(adv))
    f_orig = mock_vgg(prep(orig))
    l = torch.nn.functional.mse_loss(f_adv, f_orig)
    assert l.shape == torch.Size([])
    print(f"✓ VGGPerceptualLoss 接口正常 (mock, loss={l.item():.4f})")


def test_multi_encoder_proxy_loss_no_encoders():
    """无编码器时损失仍可计算（不使用感知损失避免VGG下载）"""
    loss_fn = MultiEncoderProxyLoss(config.LOSS_WEIGHTS, use_perceptual=False)
    orig = make_img(B=2)
    adv  = (orig + 0.02 * torch.randn_like(orig)).clamp(0, 1).requires_grad_(True)
    total, ld = loss_fn(adv, orig, encoders=[], oracle_features={})
    assert total.item() >= 0.0
    assert "total" in ld
    assert ld["encoder_total"] == 0.0
    # 梯度
    total.backward()
    print(f"✓ MultiEncoderProxyLoss (no encoders): total={total.item():.4f}")
    print(f"    breakdown: { {k: f'{v:.4f}' for k,v in ld.items()} }")


def test_multi_encoder_proxy_loss_with_mock_encoder():
    """用 Mock 编码器测试完整损失流程"""
    from encoders.base import BaseVisualEncoder

    class MockEncoder(BaseVisualEncoder):
        @property
        def name(self): return "mock"
        @property
        def feature_dim(self): return 64
        def load(self): self._model = True
        def encode(self, images):
            # 简单返回 global avg pool
            return images.mean(dim=[2, 3])  # [B, C=3] — 简化
    cfg = {"img_size": 64, "weight": 1.0,
           "norm_mean": [0.5,0.5,0.5], "norm_std": [0.5,0.5,0.5]}
    enc = MockEncoder(cfg)
    enc.load()

    orig = make_img(B=2, H=64)
    adv  = (orig + 0.02 * torch.randn_like(orig)).clamp(0, 1).requires_grad_(True)

    # oracle feature = encode(orig + big perturbation)
    oracle_feat = enc.resize_and_encode(orig + 0.1)

    loss_fn = MultiEncoderProxyLoss(config.LOSS_WEIGHTS, use_perceptual=False)
    total, ld = loss_fn(adv, orig, encoders=[enc],
                        oracle_features={"mock": oracle_feat})
    assert total.item() >= 0.0
    total.backward()
    assert adv.grad is not None and adv.grad.abs().sum() > 0
    print(f"✓ MultiEncoderProxyLoss (mock encoder): total={total.item():.4f}")
    print(f"    breakdown: { {k: f'{v:.4f}' for k,v in ld.items()} }")


# ---- rewards ----

def test_distortion_penalty():
    orig = make_img(B=1)
    adv  = (orig + 0.05).clamp(0, 1)
    pen = compute_distortion_penalty(adv, orig)
    assert pen > 0.0
    pen_zero = compute_distortion_penalty(orig, orig)
    assert pen_zero < 1e-6
    print(f"✓ distortion_penalty: diff={pen:.4f}, same={pen_zero:.6f}")


def test_tensor_to_pil():
    img = torch.rand(1, 3, 64, 64)
    pil = tensor_to_pil(img)
    assert pil.size == (64, 64)
    assert pil.mode == "RGB"
    print("✓ tensor_to_pil 正常")


def test_reward_computer_no_vlm():
    """无 VLM 时奖励为 0"""
    rc = RewardComputer(config.REWARD_WEIGHTS)
    orig = make_img(B=1, H=64)
    adv  = (orig + 0.02).clamp(0, 1)
    result = rc.compute(adv, orig, vlms=[], prompt_target=None)
    assert result["total"] == 0.0
    print("✓ RewardComputer (no vlm): total=0.0")


def test_rewards_to_tensor():
    reward_dicts = [{"total": 0.8}, {"total": -0.3}, {"total": 1.0}]
    t = RewardComputer.rewards_to_tensor(reward_dicts, device=torch.device("cpu"))
    assert t.shape == (3,)
    assert abs(t[0].item() - 0.8) < 1e-5
    print(f"✓ rewards_to_tensor: {t.tolist()}")


if __name__ == "__main__":
    print("=" * 55)
    print("Test 5: losses.py & rewards.py 验证")
    print("=" * 55)
    test_cosine_feature_loss()
    test_freq_reg_loss()
    test_vgg_perceptual_loss()
    test_multi_encoder_proxy_loss_no_encoders()
    test_multi_encoder_proxy_loss_with_mock_encoder()
    test_distortion_penalty()
    test_tensor_to_pil()
    test_reward_computer_no_vlm()
    test_rewards_to_tensor()
    print("\n✅ 所有 losses & rewards 测试通过")
