#!/bin/bash
#SBATCH -J visinject_s1a
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00
#SBATCH --output=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S1_Small_Model/logs/slurm_%j.out
#SBATCH --error=/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S1_Small_Model/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pliu07@tufts.edu

set -euo pipefail

# (可选) 课上环境模块；如果你不需要 module 也可以删掉
module load class
module load ee110/2025fall-1

# --- 你的路径（按你当前实际情况写死，避免 sbatch 环境变量不一致） ---
ENV_PATH="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject"
REPO_DIR="/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S1_Small_Model"
HF_CACHE="/cluster/tufts/c26sp1ee0141/pliu07/model_cache"

mkdir -p "${REPO_DIR}/logs"

# 激活 conda（注意：很多集群需要先 source conda.sh 才能用 conda activate）
if [ -f "${ENV_PATH}/etc/profile.d/conda.sh" ]; then
  source "${ENV_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${ENV_PATH}"

# HuggingFace 缓存（建议在脚本里再设一遍，避免非交互 shell 没读到 ~/.bashrc）
export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"

# CUDA 内存碎片优化（你指南里提到的那条）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU 线程数（跟 -c 对齐）
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "${REPO_DIR}"

# 记录环境信息（定位问题很有用）
echo "===== SLURM JOB INFO ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node : ${SLURM_NODELIST}"
echo "GPU  : ${SLURM_JOB_GPUS:-unknown}"
nvidia-smi || true
python -V
which python
echo "HF_HOME=${HF_HOME}"
echo "=========================="

# 用 srun 启动更标准（比直接 python 更好地继承资源绑定）
srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
  python run_demo.py --stage1a