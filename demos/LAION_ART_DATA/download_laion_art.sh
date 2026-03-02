#!/bin/bash
#SBATCH -J laion_art_download
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_%j.out
#SBATCH --error=/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/logs/download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pliu07@tufts.edu

set -euo pipefail

############################
# Configuration
############################
PYTHON="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject/bin/python"
DATA_DIR="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART"
LOG_DIR="${DATA_DIR}/logs"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DOWNLOADER="${SCRIPT_DIR}/download_images.py"

MODE="${1:-full}"

############################
# Modules
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# Setup
############################
mkdir -p "${LOG_DIR}"

echo "===== LAION-Art Download Job ====="
echo "JobID  : ${SLURM_JOB_ID:-local}"
echo "Node   : ${SLURM_NODELIST:-$(hostname)}"
echo "Mode   : ${MODE}"
echo "Script : ${DOWNLOADER}"
echo "=================================="

if [ ! -x "${PYTHON}" ]; then
    echo "[FATAL] Python not found: ${PYTHON}"
    exit 2
fi
${PYTHON} -V

echo "[INFO] Checking dependencies..."
${PYTHON} -c "import pyarrow; print(f'  pyarrow {pyarrow.__version__}')" || {
    echo "[WARN] Installing pyarrow..."
    "$(dirname "${PYTHON}")/pip" install pyarrow --quiet
}
${PYTHON} -c "from PIL import Image; import PIL; print(f'  Pillow {PIL.__version__}')" || {
    echo "[WARN] Installing Pillow..."
    "$(dirname "${PYTHON}")/pip" install Pillow --quiet
}
echo "[OK] Dependencies ready."

# Verify parquet files exist
PARQUET_COUNT=$(find "${DATA_DIR}/metadata" -name "*.parquet" ! -name ".*" 2>/dev/null | wc -l)
echo "[INFO] Parquet shards found: ${PARQUET_COUNT}"
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    echo "[FATAL] No parquet files in ${DATA_DIR}/metadata/"
    echo "        Download them on the login node first. See README.md."
    exit 1
fi

############################
# Run
############################
case "${MODE}" in
    test)
        echo ""
        echo "[MODE] Test: downloading 100 images"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-16} \
            ${PYTHON} "${DOWNLOADER}" \
                --test-run \
                --test-count 100 \
                --workers 8 \
                --output-dir "${DATA_DIR}/webdataset_test" \
                --log-dir "${LOG_DIR}"
        echo ""
        echo "[OK] Test done. Check logs, then run: sbatch download_laion_art.sh full"
        ;;
    full)
        echo ""
        echo "[MODE] Full download (~8M images)"
        echo "[INFO] Resumable. If it times out: sbatch download_laion_art.sh resume"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-16} \
            ${PYTHON} "${DOWNLOADER}" \
                --workers 32 \
                --output-dir "${DATA_DIR}/webdataset" \
                --log-dir "${LOG_DIR}"
        ;;
    resume)
        echo ""
        echo "[MODE] Resuming interrupted download"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-16} \
            ${PYTHON} "${DOWNLOADER}" \
                --resume \
                --workers 32 \
                --output-dir "${DATA_DIR}/webdataset" \
                --log-dir "${LOG_DIR}"
        ;;
    *)
        echo "Usage: sbatch download_laion_art.sh {test|full|resume}"
        echo ""
        echo "  test   - Download 100 images to verify setup"
        echo "  full   - Full download (~8M images)"
        echo "  resume - Continue interrupted download"
        exit 1
        ;;
esac

echo ""
echo "[DONE] Job finished at $(date)"
