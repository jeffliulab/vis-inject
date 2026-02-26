"""
Test 1: 验证 config.py 结构完整性
运行：python test/1_test_config.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config

REQUIRED_KEYS = [
    "ACTIVE_ENCODERS", "ACTIVE_PROMPT", "ACTIVE_VLMS",
    "ENCODER_CONFIG", "VLM_CONFIG", "PROMPT_CONFIG",
    "STEGO_MODEL_CONFIG", "STAGE1A_CONFIG", "STAGE1B_CONFIG",
    "STAGE2_CONFIG", "LOSS_WEIGHTS", "REWARD_WEIGHTS", "EVAL_CONFIG",
]

REQUIRED_ENCODER_KEYS = ["model_id", "img_size", "weight", "norm_mean", "norm_std", "dtype"]
REQUIRED_STEGO_KEYS   = ["base_channels", "num_res_blocks", "epsilon", "dct_patch_size",
                          "freq_band_low", "freq_band_high", "lowpass_sigma", "mode"]

def test_top_level_keys():
    for k in REQUIRED_KEYS:
        assert hasattr(config, k), f"config 缺少字段: {k}"
    print("✓ 顶层字段完整")

def test_encoder_configs():
    for name, cfg in config.ENCODER_CONFIG.items():
        for k in REQUIRED_ENCODER_KEYS:
            assert k in cfg, f"ENCODER_CONFIG['{name}'] 缺少字段: {k}"
        assert isinstance(cfg["img_size"], int) and cfg["img_size"] > 0
        assert len(cfg["norm_mean"]) == 3 and len(cfg["norm_std"]) == 3
    print(f"✓ 编码器配置完整 ({list(config.ENCODER_CONFIG.keys())})")

def test_stego_model_config():
    cfg = config.STEGO_MODEL_CONFIG
    for k in REQUIRED_STEGO_KEYS:
        assert k in cfg, f"STEGO_MODEL_CONFIG 缺少字段: {k}"
    assert cfg["epsilon"] > 0 and cfg["epsilon"] < 1
    assert cfg["freq_band_low"] < cfg["freq_band_high"]
    assert cfg["mode"] in ("fixed_token", "controllable")
    print("✓ StegoEncoder 配置合法")

def test_active_sets_valid():
    for enc in config.ACTIVE_ENCODERS:
        assert enc in config.ENCODER_CONFIG, f"ACTIVE_ENCODERS 包含未配置的编码器: {enc}"
    assert config.ACTIVE_PROMPT in config.PROMPT_CONFIG, \
        f"ACTIVE_PROMPT '{config.ACTIVE_PROMPT}' 未在 PROMPT_CONFIG 中配置"
    for vlm in config.ACTIVE_VLMS:
        assert vlm in config.VLM_CONFIG, f"ACTIVE_VLMS 包含未配置的 VLM: {vlm}"
    print(f"✓ 活跃集合有效 (encoders={config.ACTIVE_ENCODERS}, prompt={config.ACTIVE_PROMPT})")

def test_loss_weights_positive():
    for k, v in config.LOSS_WEIGHTS.items():
        assert v >= 0, f"LOSS_WEIGHTS['{k}'] 必须 >= 0"
    for k, v in config.REWARD_WEIGHTS.items():
        assert v >= 0, f"REWARD_WEIGHTS['{k}'] 必须 >= 0"
    print("✓ 损失/奖励权重非负")

if __name__ == "__main__":
    print("=" * 50)
    print("Test 1: config.py 结构验证")
    print("=" * 50)
    test_top_level_keys()
    test_encoder_configs()
    test_stego_model_config()
    test_active_sets_valid()
    test_loss_weights_positive()
    print("\n✅ 所有 config 测试通过")
