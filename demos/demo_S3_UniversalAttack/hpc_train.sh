#!/bin/bash
#SBATCH -J universal_s3
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=00-12:00:00
#SBATCH --output=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S3_UniversalAttack/logs/slurm_%j.out
#SBATCH --error=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S3_UniversalAttack/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pliu07@tufts.edu

set -euo pipefail

############################
# Configuration
############################
ENV_DIR="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject"
PYTHON="${ENV_DIR}/bin/python"
DEMO_DIR="/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S3_UniversalAttack"
LOG_DIR="${DEMO_DIR}/logs"

HF_HOME="/cluster/tufts/c26sp1ee0141/pliu07/model_cache"

# Select mode: attack | evaluate | demo
MODE="${1:-attack}"

############################
# Modules
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# Environment
############################
mkdir -p "${LOG_DIR}" "${DEMO_DIR}/checkpoints" "${DEMO_DIR}/outputs"
export HF_HOME="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== Universal Attack S3 ====="
echo "JobID : ${SLURM_JOB_ID}"
echo "Node  : ${SLURM_NODELIST}"
echo "Mode  : ${MODE}"
echo "================================"
${PYTHON} -V
nvidia-smi || true

cd "${DEMO_DIR}"

############################
# Run
############################
case "${MODE}" in
    attack)
        echo "[MODE] Single-model attack (Qwen2.5-VL-3B)"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} attack.py \
                --target-models qwen2_5_vl_3b \
                --num-steps 2000 \
                --quant-robustness
        ;;
    attack-multi)
        echo "[MODE] Multi-model attack"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} attack.py \
                --target-models qwen2_5_vl_3b qwen2_vl_2b \
                --num-steps 3000 \
                --quant-robustness \
                --multi-answer
        ;;
    evaluate)
        echo "[MODE] Evaluate universal image"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} evaluate.py \
                --image outputs/universal_final.png \
                --target-models qwen2_5_vl_3b
        ;;
    demo)
        echo "[MODE] Quick demo (500 steps)"
        srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
            ${PYTHON} demo.py --num-steps 500
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Usage: sbatch hpc_train.sh [attack|attack-multi|evaluate|demo]"
        exit 1
        ;;
esac

echo "[DONE] Mode=${MODE} finished at $(date)"
