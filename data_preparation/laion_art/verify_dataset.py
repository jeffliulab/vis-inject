"""
Verify downloaded LAION-Art WebDataset shards.

Checks:
  - Number of .tar shards and total images
  - Shard integrity (can open and list members)
  - Image readability (Pillow can decode JPEG)
  - Caption file presence
  - Reports corrupted shards and missing files

Usage:
    python verify_dataset.py
    python verify_dataset.py --data-dir /path/to/webdataset
    python verify_dataset.py --check-images    # also verify each image is decodable
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path


def verify(data_dir: str, check_images: bool = False):
    dpath = Path(data_dir)
    if not dpath.is_dir():
        print(f"[ERROR] Directory not found: {dpath}")
        sys.exit(1)

    tar_files = sorted(dpath.glob("*.tar"))
    print(f"Directory     : {dpath}")
    print(f"Shard count   : {len(tar_files)}")

    if not tar_files:
        print("[WARN] No .tar shards found.")
        return

    total_images = 0
    total_captions = 0
    total_json = 0
    corrupted_shards = []
    corrupt_images = []
    total_bytes = 0

    for tf_path in tar_files:
        total_bytes += tf_path.stat().st_size
        try:
            with tarfile.open(str(tf_path), "r") as tf:
                members = tf.getmembers()
                jpgs = [m for m in members if m.name.endswith(".jpg")]
                txts = [m for m in members if m.name.endswith(".txt")]
                jsons = [m for m in members if m.name.endswith(".json")]

                total_images += len(jpgs)
                total_captions += len(txts)
                total_json += len(jsons)

                if check_images:
                    from PIL import Image
                    import io
                    for m in jpgs:
                        try:
                            f = tf.extractfile(m)
                            Image.open(io.BytesIO(f.read())).verify()
                        except Exception:
                            corrupt_images.append(f"{tf_path.name}/{m.name}")

        except Exception as e:
            corrupted_shards.append((tf_path.name, str(e)))

    size_gb = total_bytes / (1024 ** 3)

    print(f"Total size    : {size_gb:.2f} GB")
    print(f"Total images  : {total_images:,}")
    print(f"Total captions: {total_captions:,}")
    print(f"Total metadata: {total_json:,}")

    if total_images > 0 and len(tar_files) > 0:
        print(f"Avg per shard : {total_images // len(tar_files):,}")

    if corrupted_shards:
        print(f"\n[ERROR] Corrupted shards ({len(corrupted_shards)}):")
        for name, err in corrupted_shards:
            print(f"  {name}: {err}")
    else:
        print(f"\nAll {len(tar_files)} shards OK.")

    if check_images:
        if corrupt_images:
            print(f"\n[WARN] Corrupt images ({len(corrupt_images)}):")
            for ci in corrupt_images[:20]:
                print(f"  {ci}")
            if len(corrupt_images) > 20:
                print(f"  ... and {len(corrupt_images) - 20} more")
        else:
            print(f"All {total_images:,} images verified OK.")

    # Check state file
    state_file = dpath / ".download_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"\nDownload state:")
        print(f"  Completed rows : {state.get('completed_rows', '?'):,}")
        print(f"  Successful     : {state.get('successful_downloads', '?'):,}")
        print(f"  Failed         : {state.get('failed_downloads', '?'):,}")
        sr = state.get("successful_downloads", 0) / max(state.get("completed_rows", 1), 1) * 100
        print(f"  Success rate   : {sr:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LAION-Art WebDataset")
    parser.add_argument("--data-dir", type=str,
        default="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset")
    parser.add_argument("--check-images", action="store_true",
        help="Also verify each image is decodable (slower)")
    verify(parser.parse_args().data_dir, parser.parse_args().check_images)
