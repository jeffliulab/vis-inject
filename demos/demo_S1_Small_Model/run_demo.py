"""
demo_S1_Small_Model 主入口脚本

用法：
  python run_demo.py --stage1a                        # Stage 1A：单编码器快速验证
  python run_demo.py --stage1b                        # Stage 1B：多编码器完整训练
  python run_demo.py --stage2 --checkpoint PATH       # Stage 2：REINFORCE RL 微调
  python run_demo.py --eval --checkpoint PATH         # 评估模型
  python run_demo.py --infer --checkpoint PATH --image PATH  # 单张图像推理

可选覆盖 config：
  --encoders blip2,qwen    覆盖 ACTIVE_ENCODERS
  --prompt fixed_keyword   覆盖 ACTIVE_PROMPT
  --vlms qwen,deepseek     覆盖 ACTIVE_VLMS
  --num-images 200         覆盖图像数量
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="demo_S1 StegoEncoder 主入口")

    # 模式选择
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage1a", action="store_true", help="Stage 1A：单编码器快速验证")
    mode.add_argument("--stage1b", action="store_true", help="Stage 1B：多编码器训练")
    mode.add_argument("--stage2",  action="store_true", help="Stage 2：REINFORCE RL 微调")
    mode.add_argument("--eval",    action="store_true", help="评估模型")
    mode.add_argument("--infer",   action="store_true", help="单张图像推理")
    mode.add_argument("--list",    action="store_true", help="列出已注册的编码器/VLM/Prompt")

    # 公共参数
    p.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    p.add_argument("--image",      type=str, default=None, help="输入图像路径（--infer 模式）")
    p.add_argument("--output",     type=str, default=None, help="输出图像路径（--infer 模式）")

    # Config 覆盖
    p.add_argument("--encoders",   type=str, default=None, help="覆盖 ACTIVE_ENCODERS（逗号分隔）")
    p.add_argument("--prompt",     type=str, default=None, help="覆盖 ACTIVE_PROMPT")
    p.add_argument("--vlms",       type=str, default=None, help="覆盖 ACTIVE_VLMS（逗号分隔）")
    p.add_argument("--num-images", type=int, default=None, help="覆盖训练/评估图像数量")
    p.add_argument("--device",     type=str, default=None, help="覆盖 DEVICE (cuda/cpu)")

    return p.parse_args()


def apply_overrides(args):
    """将命令行参数覆盖到 config"""
    import config as cfg
    if args.encoders:
        cfg.ACTIVE_ENCODERS = [e.strip() for e in args.encoders.split(",")]
        print(f"[Override] ACTIVE_ENCODERS = {cfg.ACTIVE_ENCODERS}")
    if args.prompt:
        cfg.ACTIVE_PROMPT = args.prompt
        print(f"[Override] ACTIVE_PROMPT = {cfg.ACTIVE_PROMPT}")
    if args.vlms:
        cfg.ACTIVE_VLMS = [v.strip() for v in args.vlms.split(",")]
        print(f"[Override] ACTIVE_VLMS = {cfg.ACTIVE_VLMS}")
    if args.device:
        cfg.DEVICE = args.device
        print(f"[Override] DEVICE = {cfg.DEVICE}")
    if args.num_images:
        for stage_cfg in [cfg.STAGE1A_CONFIG, cfg.STAGE1B_CONFIG]:
            stage_cfg["num_images"] = args.num_images
        cfg.EVAL_CONFIG["num_test_images"] = args.num_images
        print(f"[Override] num_images = {args.num_images}")


def mode_list():
    """列出所有已注册的组件"""
    from encoders import list_encoders
    from vlms import list_vlms
    from prompts import list_prompts
    import config as cfg

    print("\n=== 已注册编码器 ===")
    for name in list_encoders():
        enc_cfg = cfg.ENCODER_CONFIG.get(name, {})
        print(f"  [{name}] img_size={enc_cfg.get('img_size','?')} "
              f"model_id={enc_cfg.get('model_id','?')}")

    print("\n=== 已注册 VLM ===")
    for name in list_vlms():
        vlm_cfg = cfg.VLM_CONFIG.get(name, {})
        print(f"  [{name}] model_id={vlm_cfg.get('model_id','?')}")

    print("\n=== 已注册 Prompt ===")
    from prompts import load_prompt
    for name in list_prompts():
        pt = load_prompt(name, cfg.PROMPT_CONFIG)
        print(f"  [{name}] target='{pt.target_text[:60]}...'")

    print(f"\n当前活跃配置:")
    print(f"  ACTIVE_ENCODERS = {cfg.ACTIVE_ENCODERS}")
    print(f"  ACTIVE_PROMPT   = {cfg.ACTIVE_PROMPT}")
    print(f"  ACTIVE_VLMS     = {cfg.ACTIVE_VLMS}")


def mode_infer(args):
    """单张图像推理"""
    import torch
    import config as cfg
    from models.stego_encoder import StegoEncoder
    from utils import load_image, pil_to_tensor, tensor_to_pil, save_image, calculate_psnr

    if not args.checkpoint:
        print("错误：--infer 模式需要 --checkpoint 参数")
        sys.exit(1)
    if not args.image:
        print("错误：--infer 模式需要 --image 参数")
        sys.exit(1)

    # 加载模型
    model = StegoEncoder(cfg.STEGO_MODEL_CONFIG).to(cfg.DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"模型加载完成: {args.checkpoint}")

    # 推理
    orig_pil = load_image(args.image, size=(256, 256))
    orig_t   = pil_to_tensor(orig_pil, cfg.DEVICE)

    with torch.no_grad():
        adv_t = model(orig_t)

    psnr = calculate_psnr(orig_t.cpu(), adv_t.cpu())
    delta_max = (adv_t - orig_t).abs().max().item()
    print(f"推理完成: PSNR={psnr:.1f}dB, L∞={delta_max:.4f}")

    output_path = args.output or args.image.replace(".", "_adv.")
    save_image(adv_t, output_path)
    print(f"对抗图像已保存: {output_path}")


def main():
    args = parse_args()
    apply_overrides(args)

    if args.list:
        mode_list()

    elif args.stage1a:
        # 使用端到端训练（E2E）：直接通过 Qwen CE loss 训练 StegoEncoder
        # 比代理特征损失 (proxy feature loss) 更直接，训练信号更强
        from training.e2e_trainer import E2ETrainer
        trainer = E2ETrainer(stage="1a")
        trainer.run()

    elif args.stage1b:
        from training.proxy_trainer import ProxyTrainer
        trainer = ProxyTrainer(stage="1b")
        trainer.run()

    elif args.stage2:
        if not args.checkpoint:
            print("错误：--stage2 需要 --checkpoint 参数")
            sys.exit(1)
        from training.rl_trainer import RLTrainer
        trainer = RLTrainer(args.checkpoint)
        trainer.run()

    elif args.eval:
        if not args.checkpoint:
            print("错误：--eval 需要 --checkpoint 参数")
            sys.exit(1)
        from evaluate import Evaluator
        evaluator = Evaluator(args.checkpoint)
        evaluator.run()

    elif args.infer:
        mode_infer(args)


if __name__ == "__main__":
    main()
