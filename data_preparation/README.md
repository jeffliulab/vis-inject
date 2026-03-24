# Data Preparation

Scripts for downloading and preparing all data needed by the VisInject pipeline.

## Directory Structure

```
data_preparation/
├── laion_art/                    # LAION-Art dataset (~8M art images)
│   ├── download_parquet_metadata.sh  # Step 1: metadata (login node)
│   ├── download_images.py            # Step 2: images (compute node)
│   ├── download_laion_art.sh         # SLURM wrapper for Step 2
│   └── verify_dataset.py            # Step 3: verification
├── models/
│   ├── download_all_models.py        # Download HF VLMs for pipeline
│   └── download_decoder_weights.py   # Download AnyAttack decoder
└── demo_images/
    └── upload_demo_images.sh         # Upload test images to HPC
```

## Quick Start (HPC)

### 1. Download VLM models

```bash
# On login node (has internet):
python data_preparation/models/download_all_models.py --stage quick   # 3 VLMs + CLIP (~16GB)
python data_preparation/models/download_all_models.py --stage full    # 5 VLMs + CLIP (~38GB)
```

### 2. Download AnyAttack decoder weights

```bash
python data_preparation/models/download_decoder_weights.py
```

### 3. Upload demo images

```bash
bash data_preparation/demo_images/upload_demo_images.sh
```

### 4. (Optional) Download LAION-Art for self-training

Only needed if you want to train your own AnyAttack decoder (demo_S2).
The pipeline works with the pretrained `coco_bi.pt` weights.

```bash
# Step 1: Parquet metadata (login node, ~1.3 GB)
bash data_preparation/laion_art/download_parquet_metadata.sh

# Step 2: Images (compute node, ~80-150 GB)
sbatch data_preparation/laion_art/download_laion_art.sh test    # verify setup
sbatch data_preparation/laion_art/download_laion_art.sh full    # full download
sbatch data_preparation/laion_art/download_laion_art.sh resume  # if interrupted

# Step 3: Verify
python data_preparation/laion_art/verify_dataset.py
```

## HPC Paths

| Resource | Path |
|----------|------|
| Model cache | `/cluster/tufts/c26sp1ee0141/pliu07/model_cache` |
| LAION metadata | `/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/metadata` |
| LAION images | `/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset` |
| Conda env | `/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject` |
| Decoder weights | `demos/demo_S2P/checkpoints/coco_bi.pt` |
