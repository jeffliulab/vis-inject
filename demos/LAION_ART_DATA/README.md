# LAION-Art Dataset Download

Downloads LAION-Art (~8M art images) for AnyAttack (demo_S2) pre-training.

## Architecture

```
HPC Login Node                          HPC Compute Node (batch job)
─────────────                           ────────────────────────────
1. huggingface-cli login                3. sbatch download_laion_art.sh test
2. download_parquet_metadata.sh         4. sbatch download_laion_art.sh full
   (128 parquet shards → metadata/)     5. sbatch download_laion_art.sh resume
                                           ↓
                                        download_images.py
                                           reads metadata/*.parquet
                                           downloads images from source URLs
                                           resizes to 224x224, center crop
                                           writes WebDataset .tar shards
                                           ↓
                                        webdataset/00000.tar, 00001.tar, ...
```

**Why two steps?**  Compute nodes on Tufts HPC cannot resolve
`cdn-lfs.huggingface.co` (DNS blocked), so parquet metadata must be
downloaded on the login node. Image URLs point to external hosts which
compute nodes *can* access.

## Prerequisites

```bash
# Python packages (already in visinject conda env)
pip install pyarrow Pillow

# HuggingFace authentication (one-time, on login node)
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN_HERE" > ~/.cache/huggingface/token

# Get your token at: https://huggingface.co/settings/tokens
# Accept dataset terms at: https://huggingface.co/datasets/laion/laion-art
```

## Step 1: Download Parquet Metadata (Login Node)

The dataset metadata is split into 128 parquet shards (~1.3 GB total).
Run this on the **login node** (has full internet access):

```bash
cd /path/to/demos/LAION_ART_DATA
bash download_parquet_metadata.sh
```

Or manually:

```bash
/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject/bin/python -c "
from huggingface_hub import HfApi, hf_hub_download
import os
token = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
api = HfApi(token=token)
out_dir = '/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/metadata'
os.makedirs(out_dir, exist_ok=True)
files = api.list_repo_files('laion/laion-art', repo_type='dataset')
parquets = [f for f in files if f.endswith('.parquet') and not f.startswith('.')]
print(f'Found {len(parquets)} parquet files')
for i, pf in enumerate(parquets):
    print(f'[{i+1}/{len(parquets)}] {pf}')
    hf_hub_download('laion/laion-art', repo_type='dataset',
                    filename=pf, local_dir=out_dir, token=token)
print('Done!')
"
```

Use `screen` or `tmux` to prevent SSH disconnection:
```bash
screen -S parquet_download
# run the command above
# Ctrl+A, D to detach; screen -r parquet_download to reattach
```

## Step 2: Test Image Download (Compute Node)

```bash
cd /path/to/demos/LAION_ART_DATA
sbatch download_laion_art.sh test
```

This downloads 100 images to verify the setup works. Check the output:
```bash
cat /cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_<JOBID>.out
```

## Step 3: Full Download

```bash
sbatch download_laion_art.sh full
```

## Step 4: Resume (if job times out)

The downloader saves progress to `.download_state.json`. Simply resubmit:

```bash
sbatch download_laion_art.sh resume
```

Repeat until complete.

## Step 5: Verify

```bash
python verify_dataset.py
python verify_dataset.py --check-images   # thorough check (slower)
```

## File Structure

```
LAION_ART_DATA/
├── download_images.py          # Main downloader (no img2dataset needed)
├── download_laion_art.sh       # SLURM batch script
├── download_parquet_metadata.sh # Parquet download helper (login node)
├── download_dataset_on_hpc     # Quick-reference parquet download command
├── verify_dataset.py           # Post-download verification
└── README.md

/cluster/.../LAION_ART/
├── metadata/                   # 128 parquet shards (pre-downloaded)
│   ├── part-00000-*.snappy.parquet
│   ├── part-00001-*.snappy.parquet
│   └── ...
├── webdataset/                 # Downloaded images (WebDataset format)
│   ├── 00000.tar
│   ├── 00001.tar
│   ├── ...
│   └── .download_state.json
├── webdataset_test/            # Test download output
└── logs/
    ├── download_YYYYMMDD_HHMMSS.log      # Full run log
    ├── failed_urls_YYYYMMDD_HHMMSS.log   # Every failed URL + reason
    └── error_snapshot_YYYYMMDD_HHMMSS.json # Error category counts
```

## Log Files

Each run generates comprehensive logs for debugging:

| File | Content |
|------|---------|
| `download_*.log` | Full log: system info, SLURM env, config, progress, final report |
| `failed_urls_*.log` | One line per failed URL with error category and row index |
| `error_snapshot_*.json` | JSON: error counts by category + last 50 error details |
| `.download_state.json` | Resume state: completed rows, success/fail counts, session history |

Error categories tracked: `timeout`, `dns_failure`, `connection_refused`,
`connection_reset`, `ssl_error`, `http_404`, `http_403`, `http_other`,
`invalid_url`, `empty_response`, `image_corrupt`, `image_too_small`,
`pillow_error`, `unknown`.

## Troubleshooting

**Q: "No parquet files found"**
A: You need to download parquet metadata on the login node first (Step 1).
Compute nodes cannot access HuggingFace CDN.

**Q: Very low success rate?**
A: LAION URLs are from 2022; many hosts are gone. 40-60% success rate is
normal. Check `failed_urls_*.log` for the dominant error category.

**Q: Job timed out before finishing?**
A: Just run `sbatch download_laion_art.sh resume`. Progress is saved
automatically.

**Q: How much disk space needed?**
A: ~80-150 GB for the images (depends on success rate). Parquet metadata
is ~1.3 GB. Check with `df -h /cluster/tufts/c26sp1ee0141/pliu07/`.

**Q: Download is slow?**
A: Adjust `--workers` in the SLURM script. Default is 32 threads. On
slow networks try 8-16; on fast networks try 48-64.
