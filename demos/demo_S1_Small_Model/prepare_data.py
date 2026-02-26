"""
数据准备脚本 — 下载 COCO val2017 并整理到 data/train/ 和 data/test/

默认行为（直接运行 python prepare_data.py）：
  下载 COCO val2017 全量（5000张，约1GB），4000张训练 + 500张测试

常用命令：
  python prepare_data.py                          # 下载全量 COCO（推荐）
  python prepare_data.py --num-train 200          # 只取 200 张，Stage 1A 快速验证
  python prepare_data.py --source synthetic       # 合成图像，完全离线，用于代码调试
  python prepare_data.py --source local --image-dir /path/to/images  # 使用本地图像集

支持的数据源：
  coco       : HuggingFace datasets 下载 COCO 2017 val（联网，~1GB，推荐）
  local      : 从本地目录复制图像（已有数据集时使用）
  synthetic  : 生成合成随机图像（无需下载，仅用于代码流程调试）
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# 图像预处理：保持比例 + 中心裁剪
# ============================================================

def center_crop_resize(img: Image.Image, size: int = 512) -> Image.Image:
    """
    保持宽高比，先缩放短边到 size，再中心裁剪到 size×size。
    比直接 resize(size, size) 更好，不会拉伸变形。
    """
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


# ============================================================
# 数据源：COCO
# ============================================================

def download_coco(num_train: int, num_test: int, output_dir: str):
    """
    直接从 COCO 官方服务器下载 val2017 zip（~1GB，5000张），
    比通过 HuggingFace datasets 快得多（HF 需要扫描全部 40 个分片）。
    """
    import urllib.request
    import zipfile
    import tempfile

    COCO_VAL_URL  = "http://images.cocodataset.org/zips/val2017.zip"
    total_needed  = num_train + num_test

    output_dir  = Path(output_dir)
    cache_zip   = output_dir / "val2017.zip"
    cache_imgs  = output_dir / "val2017_extracted"
    train_dir   = output_dir / "train"
    test_dir    = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # ---- 下载 zip（若已缓存则跳过）----
    if not cache_zip.exists():
        print(f"正在从 COCO 官方服务器下载 val2017.zip ...")
        print(f"URL: {COCO_VAL_URL}")
        print(f"预计大小: ~1GB，请耐心等待...\n")

        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                pct = block_num * block_size / total_size * 100
                done = block_num * block_size / (1024 * 1024)
                total = total_size / (1024 * 1024)
                print(f"\r  已下载: {done:.0f} MB / {total:.0f} MB ({pct:.1f}%)",
                      end="", flush=True)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(COCO_VAL_URL, cache_zip, _reporthook)
            print(f"\n  ✓ 下载完成: {cache_zip}")
        except Exception as e:
            print(f"\n  下载失败: {e}")
            if cache_zip.exists():
                cache_zip.unlink()
            return False
    else:
        print(f"✓ 发现缓存: {cache_zip}，跳过下载")

    # ---- 解压 zip ----
    if not cache_imgs.exists():
        print(f"正在解压 val2017.zip ...")
        try:
            with zipfile.ZipFile(cache_zip, "r") as zf:
                zf.extractall(cache_imgs)
            print(f"  ✓ 解压完成")
        except Exception as e:
            print(f"  解压失败: {e}")
            return False
    else:
        print(f"✓ 发现已解压目录: {cache_imgs}")

    # ---- 找到所有图像 ----
    img_dir = cache_imgs / "val2017"
    if not img_dir.exists():
        # 有时解压后直接在 cache_imgs 下
        img_dir = cache_imgs
    all_imgs = sorted([p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"共找到 {len(all_imgs)} 张图像")

    if len(all_imgs) < total_needed:
        print(f"警告：图像数量（{len(all_imgs)}）少于请求量（{total_needed}），"
              f"将使用全部图像")
        total_needed = len(all_imgs)
        num_train = int(total_needed * 0.89)
        num_test  = total_needed - num_train

    random.shuffle(all_imgs)
    train_imgs = all_imgs[:num_train]
    test_imgs  = all_imgs[num_train:num_train + num_test]

    # ---- 处理并保存 ----
    print(f"处理训练集（{len(train_imgs)} 张）...")
    _process_and_save(train_imgs, train_dir)

    print(f"处理测试集（{len(test_imgs)} 张）...")
    _process_and_save(test_imgs, test_dir)

    return True


def _process_and_save(src_paths: list, dest_dir: Path):
    for i, src in enumerate(tqdm(src_paths)):
        try:
            img = Image.open(src).convert("RGB")
            img = center_crop_resize(img, size=512)
            img.save(dest_dir / f"{i:05d}.jpg", quality=95)
        except Exception as e:
            print(f"  跳过 {src.name}: {e}")


# ============================================================
# 数据源：本地目录
# ============================================================

def prepare_from_local(image_dir: str, num_train: int, num_test: int,
                        output_dir: str):
    """从本地图像目录复制并预处理图像"""
    src = Path(image_dir)
    if not src.exists():
        print(f"错误：目录不存在 {image_dir}")
        return False

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    if not all_paths:
        print(f"错误：{image_dir} 中未找到图像文件")
        return False

    print(f"找到 {len(all_paths)} 张图像")
    random.shuffle(all_paths)

    train_dir = Path(output_dir) / "train"
    test_dir  = Path(output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"处理训练集（{num_train} 张）...")
    _process_and_save(all_paths[:num_train], train_dir)
    print(f"处理测试集（{num_test} 张）...")
    _process_and_save(all_paths[num_train:num_train + num_test], test_dir)
    return True


# ============================================================
# 数据源：合成随机图像（仅用于代码调试）
# ============================================================

def prepare_synthetic(num_train: int, num_test: int, output_dir: str):
    """生成合成随机图像（彩色 + 随机形状，比纯噪声更接近真实图像分布）"""
    import random as rnd

    train_dir = Path(output_dir) / "train"
    test_dir  = Path(output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def make_synthetic_image(seed: int, size: int = 256) -> Image.Image:
        rnd.seed(seed)
        # 随机背景色
        bg = tuple(rnd.randint(50, 200) for _ in range(3))
        img = Image.new("RGB", (size, size), bg)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # 随机矩形/椭圆
        for _ in range(rnd.randint(3, 8)):
            x1, y1 = rnd.randint(0, size//2), rnd.randint(0, size//2)
            x2, y2 = rnd.randint(size//2, size), rnd.randint(size//2, size)
            color = tuple(rnd.randint(0, 255) for _ in range(3))
            if rnd.random() > 0.5:
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color)
        return img

    print(f"生成合成训练图像（{num_train} 张）...")
    for i in tqdm(range(num_train)):
        img = make_synthetic_image(seed=i, size=256)
        img.save(train_dir / f"{i:05d}.jpg", quality=90)

    print(f"生成合成测试图像（{num_test} 张）...")
    for i in tqdm(range(num_test)):
        img = make_synthetic_image(seed=num_train + i, size=256)
        img.save(test_dir / f"{i:05d}.jpg", quality=90)

    return True


# ============================================================
# 验证数据集
# ============================================================

def verify_dataset(output_dir: str):
    """验证数据集目录结构和图像完整性"""
    output_dir = Path(output_dir)
    issues = []

    for split in ["train", "test"]:
        split_dir = output_dir / split
        if not split_dir.exists():
            issues.append(f"  缺少目录: {split_dir}")
            continue

        imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        if not imgs:
            issues.append(f"  {split}/ 目录为空")
            continue

        # 验证前 5 张可读
        errors = 0
        for p in imgs[:5]:
            try:
                img = Image.open(p)
                img.verify()
            except Exception:
                errors += 1
        print(f"  {split}/: {len(imgs)} 张图像，抽样验证 {'✓' if errors == 0 else f'✗ ({errors} 张损坏)'}")

    if issues:
        print("\n警告：")
        for issue in issues:
            print(issue)
        return False

    print("\n✅ 数据集验证通过")
    return True


# ============================================================
# 主入口
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="demo_S1 数据准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python prepare_data.py                          # 默认：下载 COCO val2017 全量（~1GB）
  python prepare_data.py --num-train 200          # 快速验证：只用 200 张 COCO 图像
  python prepare_data.py --source synthetic       # 离线调试：合成随机图（秒级完成）
  python prepare_data.py --source local --image-dir D:/my/images  # 使用本地图像集

关于 COCO 数据集：
  COCO val2017 共 5000 张高质量日常照片（猫狗、车、人、风景等），约 1GB。
  本项目只使用图像本身，不需要 COCO 的检测框/分割标注。
  图像内容多样，是训练通用 StegoEncoder 的理想数据源。
        """,
    )
    p.add_argument("--source", choices=["coco", "local", "synthetic"],
                   default="coco",
                   help="数据来源 (default: coco，下载 COCO val2017)")
    p.add_argument("--image-dir", type=str, default=None,
                   help="本地图像目录（--source local 时使用）")
    p.add_argument("--num-train", type=int, default=4000,
                   help="训练图像数量 (default: 4000，Stage 1A 建议 200)")
    p.add_argument("--num-test",  type=int, default=500,
                   help="测试/评估图像数量 (default: 500)")
    p.add_argument("--output-dir", type=str, default="data",
                   help="输出目录 (default: data/)")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    args = p.parse_args()

    random.seed(args.seed)

    print(f"数据准备：source={args.source}, train={args.num_train}, test={args.num_test}")
    print(f"输出目录：{args.output_dir}/")

    if args.source == "coco":
        ok = download_coco(args.num_train, args.num_test, args.output_dir)
    elif args.source == "local":
        if not args.image_dir:
            print("错误：--source local 需要指定 --image-dir")
            return
        ok = prepare_from_local(args.image_dir, args.num_train,
                                 args.num_test, args.output_dir)
    else:  # synthetic
        ok = prepare_synthetic(args.num_train, args.num_test, args.output_dir)

    if ok:
        print("\n验证数据集...")
        verify_dataset(args.output_dir)
        print(f"\n完成！可用的训练命令：")
        print(f"  # Stage 1A（4090 单编码器快速验证，~1-2小时）")
        print(f"  python run_demo.py --stage1a --encoders qwen --num-images 200")
        print(f"")
        print(f"  # Stage 1B（4090 多编码器完整训练，~4-6小时）")
        print(f"  python run_demo.py --stage1b --encoders blip2,deepseek,qwen --num-images {args.num_train}")
    else:
        print("\n数据准备失败，请检查上方错误信息")


if __name__ == "__main__":
    main()
