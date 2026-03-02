#!/bin/bash
# Download LAION-Art parquet metadata on the LOGIN NODE.
# Compute nodes cannot resolve cdn-lfs.huggingface.co, so this
# must be run where full internet access is available.
#
# Usage (on login node):
#   bash download_parquet_metadata.sh
#
# Use screen/tmux if your SSH session might disconnect:
#   screen -S parquet && bash download_parquet_metadata.sh

set -euo pipefail

PYTHON="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject/bin/python"
OUT_DIR="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/metadata"

if [ ! -f "${HOME}/.cache/huggingface/token" ]; then
    echo "[ERROR] HuggingFace token not found."
    echo "  Create one at https://huggingface.co/settings/tokens"
    echo "  Then run: mkdir -p ~/.cache/huggingface && echo 'hf_xxx' > ~/.cache/huggingface/token"
    exit 1
fi

echo "[INFO] Downloading LAION-Art parquet metadata to ${OUT_DIR}"
echo "[INFO] This will download 128 parquet shards (~1.3 GB total)"
echo ""

${PYTHON} -c "
from huggingface_hub import HfApi, hf_hub_download
import os

token = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
api = HfApi(token=token)
out_dir = '${OUT_DIR}'
os.makedirs(out_dir, exist_ok=True)

files = api.list_repo_files('laion/laion-art', repo_type='dataset')
parquets = [f for f in files if f.endswith('.parquet') and not f.startswith('.')]
print(f'Found {len(parquets)} parquet files')

for i, pf in enumerate(parquets):
    dest = os.path.join(out_dir, os.path.basename(pf))
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f'[{i+1}/{len(parquets)}] SKIP (exists) {pf}')
        continue
    print(f'[{i+1}/{len(parquets)}] Downloading {pf}...')
    hf_hub_download(
        repo_id='laion/laion-art',
        repo_type='dataset',
        filename=pf,
        local_dir=out_dir,
        token=token
    )

print()
print('Done! Parquet files saved to:', out_dir)
count = len([f for f in os.listdir(out_dir) if f.endswith('.parquet')])
print(f'Total parquet files: {count}')
"
