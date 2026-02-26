"""
Test 6: 集成测试 — 端到端训练循环（小规模 dry run，不加载大模型）
验证：config → StegoEncoder → 损失计算 → 梯度更新 → checkpoint 保存
运行：conda run -n deeplearning python test/6_test_integration.py
"""
import sys, os, tempfile, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import config as cfg

# 临时覆盖 config 为最小值
cfg.ACTIVE_ENCODERS = []   # 不加载任何编码器（dry run）
cfg.ACTIVE_VLMS     = []
cfg.STAGE1A_CONFIG["epochs"]     = 2
cfg.STAGE1A_CONFIG["num_images"] = 4
cfg.STAGE1A_CONFIG["batch_size"] = 2
cfg.STAGE1A_CONFIG["oracle_pgd_steps"] = 5
cfg.STEGO_MODEL_CONFIG["base_channels"]  = 16   # 最小模型，节省内存
cfg.STEGO_MODEL_CONFIG["num_res_blocks"] = 1


def test_full_training_step():
    """完整训练步骤：前向 → 损失 → 反传 → 参数更新"""
    from models.stego_encoder import StegoEncoder
    from losses import MultiEncoderProxyLoss
    from augmentation import DifferentiableAugmentor

    model = StegoEncoder(cfg.STEGO_MODEL_CONFIG)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = MultiEncoderProxyLoss(cfg.LOSS_WEIGHTS, use_perceptual=False)
    augmentor = DifferentiableAugmentor()

    orig = torch.rand(2, 3, 64, 64)

    # 前向
    adv = model(orig)
    assert adv.shape == orig.shape

    # 损失（无编码器）
    total, ld = loss_fn(adv, orig, encoders=[], oracle_features={}, augmentor=augmentor)
    assert total.item() >= 0.0

    # 反传
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    # 低通平滑
    with torch.no_grad():
        smoothed = model.apply_lowpass_smoothing(adv.detach(), orig)

    print(f"✓ 完整训练步骤: loss={total.item():.4f}, PSNR={-10*torch.log10(((adv-orig)**2).mean()):.1f}dB")


def test_checkpoint_save_load():
    """checkpoint 保存和加载"""
    from models.stego_encoder import StegoEncoder

    tmpdir = tempfile.mkdtemp()
    try:
        model = StegoEncoder(cfg.STEGO_MODEL_CONFIG)
        ckpt_path = os.path.join(tmpdir, "test.pt")
        torch.save({"epoch": 1, "model_state": model.state_dict(),
                    "loss": 0.5, "config": cfg.STEGO_MODEL_CONFIG}, ckpt_path)

        # 加载
        model2 = StegoEncoder(cfg.STEGO_MODEL_CONFIG)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt["model_state"])
        assert ckpt["epoch"] == 1

        # 验证参数一致
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
        print(f"✓ checkpoint 保存/加载正确 ({ckpt_path})")
    finally:
        shutil.rmtree(tmpdir)


def test_run_demo_list():
    """run_demo.py --list 应正常运行"""
    import subprocess
    result = subprocess.run(
        ["conda", "run", "-n", "deeplearning", "python", "run_demo.py", "--list"],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    assert result.returncode == 0, f"--list 失败:\n{result.stderr}"
    assert "已注册编码器" in result.stdout
    assert "blip2" in result.stdout
    print("✓ run_demo.py --list 正常")
    print(result.stdout[:300])


def test_run_demo_help():
    """run_demo.py --help 应正常输出"""
    import subprocess
    result = subprocess.run(
        ["conda", "run", "-n", "deeplearning", "python", "run_demo.py", "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    assert result.returncode == 0
    assert "stage1a" in result.stdout
    print("✓ run_demo.py --help 正常")


def test_proxy_trainer_dry_run():
    """ProxyTrainer dry run（合成图像，无编码器）"""
    from training.proxy_trainer import ProxyTrainer

    # 使用 dry run 配置（已在文件顶部覆盖）
    trainer = ProxyTrainer(stage="1a")
    model, history = trainer.run()

    assert model is not None
    assert len(history) == cfg.STAGE1A_CONFIG["epochs"]
    assert all("loss" in h for h in history)
    print(f"✓ ProxyTrainer dry run 完成: {len(history)} epochs")
    print(f"  最终 loss: {history[-1]['loss']:.4f}")


if __name__ == "__main__":
    print("=" * 55)
    print("Test 6: 集成测试（dry run，无大模型）")
    print("=" * 55)
    test_full_training_step()
    test_checkpoint_save_load()
    test_run_demo_help()
    test_run_demo_list()
    test_proxy_trainer_dry_run()
    print("\n✅ 所有集成测试通过")
