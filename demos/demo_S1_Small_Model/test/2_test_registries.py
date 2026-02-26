"""
Test 2: 验证三个注册表（encoders/vlms/prompts）的注册和实例化
注意：不实际加载模型权重，只验证接口和注册机制
运行：python test/2_test_registries.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config

# ---- 1. 编码器注册表 ----
def test_encoder_registry():
    from encoders import ENCODER_REGISTRY, list_encoders, load_encoders
    print(f"已注册编码器: {list_encoders()}")
    expected = {"blip2", "deepseek", "qwen"}
    assert expected.issubset(set(list_encoders())), \
        f"缺少编码器: {expected - set(list_encoders())}"

    # 实例化（不加载权重）
    encoders = load_encoders(["blip2", "deepseek", "qwen"], config.ENCODER_CONFIG)
    assert len(encoders) == 3

    for enc in encoders:
        assert hasattr(enc, "name")
        assert hasattr(enc, "img_size")
        assert hasattr(enc, "norm_mean")
        assert hasattr(enc, "norm_std")
        assert hasattr(enc, "feature_dim")
        assert hasattr(enc, "encode")
        assert hasattr(enc, "resize_and_encode")
        assert hasattr(enc, "normalize")
        assert enc.img_size > 0
        assert len(enc.norm_mean) == 3
        print(f"  ✓ {enc}")

    print("✓ 编码器注册表正常")


# ---- 2. VLM 注册表 ----
def test_vlm_registry():
    from vlms import VLM_REGISTRY, list_vlms, load_vlms
    print(f"已注册 VLM: {list_vlms()}")
    expected = {"blip2", "deepseek", "qwen"}
    assert expected.issubset(set(list_vlms())), \
        f"缺少 VLM: {expected - set(list_vlms())}"

    vlms = load_vlms(["blip2", "deepseek", "qwen"], config.VLM_CONFIG)
    assert len(vlms) == 3

    for vlm in vlms:
        assert hasattr(vlm, "name")
        assert hasattr(vlm, "generate")
        assert hasattr(vlm, "load")
        assert not vlm.is_loaded(), "VLM 不应在实例化时自动加载权重"
        print(f"  ✓ {vlm}")

    print("✓ VLM 注册表正常")


# ---- 3. Prompt 注册表 ----
def test_prompt_registry():
    from prompts import PROMPT_REGISTRY, list_prompts, load_prompt
    print(f"已注册 Prompt: {list_prompts()}")
    expected = {"fixed_keyword", "harry_potter_style", "ignore_previous"}
    assert expected.issubset(set(list_prompts())), \
        f"缺少 Prompt: {expected - set(list_prompts())}"

    for name in expected:
        pt = load_prompt(name, config.PROMPT_CONFIG)
        assert pt.name == name
        assert isinstance(pt.target_text, str) and len(pt.target_text) > 0
        # 测试 compute_success
        score_hit  = pt.compute_success(pt.target_text)   # 包含目标文字
        score_miss = pt.compute_success("完全无关的普通回答xyz")
        assert 0.0 <= score_hit  <= 1.0
        assert 0.0 <= score_miss <= 1.0
        print(f"  ✓ {pt.name}: hit={score_hit:.2f}, miss={score_miss:.2f}")

    print("✓ Prompt 注册表正常")


# ---- 4. 重复注册应报错 ----
def test_duplicate_registration():
    from encoders.registry import register_encoder, ENCODER_REGISTRY
    try:
        @register_encoder("blip2")
        class DuplicateEncoder:
            pass
        assert False, "重复注册应抛出 ValueError"
    except ValueError as e:
        print(f"✓ 重复注册正确报错: {e}")


# ---- 5. 归一化工具测试 ----
def test_normalize_utility():
    import torch
    from encoders import load_encoders
    encoders = load_encoders(["qwen"], config.ENCODER_CONFIG)
    enc = encoders[0]
    img = torch.rand(2, 3, 392, 392)  # [B, C, H, W], [0,1]
    normed = enc.normalize(img)
    assert normed.shape == img.shape
    # 归一化后均值应接近 0
    assert normed.mean().abs() < 0.5, "归一化后均值应接近 0"
    print(f"✓ 归一化工具正常 (output mean={normed.mean():.3f})")


if __name__ == "__main__":
    print("=" * 50)
    print("Test 2: 注册表机制验证")
    print("=" * 50)
    test_encoder_registry()
    test_vlm_registry()
    test_prompt_registry()
    test_duplicate_registration()
    test_normalize_utility()
    print("\n✅ 所有注册表测试通过")
