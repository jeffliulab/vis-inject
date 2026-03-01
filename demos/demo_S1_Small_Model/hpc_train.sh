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

############################
# 0) 绝对路径配置（按你当前实际环境）
############################
ENV_DIR="/cluster/tufts/c26sp1ee0141/pliu07/condaenv/visinject"
PYTHON="${ENV_DIR}/bin/python"

DEMO_DIR="/cluster/tufts/c26sp1ee0141/pliu07/vis_inject/demos/demo_S1_Small_Model"
LOG_DIR="${DEMO_DIR}/logs"

HF_HOME="/cluster/tufts/c26sp1ee0141/pliu07/model_cache"
HF_HUB_CACHE="${HF_HOME}/hub"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
HF_DATASETS_CACHE="${HF_HOME}/datasets"

############################
# 1) 可选：模块（如果你们集群跑 Python/GPU 必须加载，就保留）
############################
module load class || true
module load ee110/2025fall-1 || true

############################
# 2) 目录准备
############################
mkdir -p "${LOG_DIR}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

############################
# 3) 环境变量（全部在脚本内设置，避免 sbatch 不读 ~/.bashrc）
############################
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"

# 关键：CUDA 内存碎片优化（你指南里提到的）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 线程数对齐
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

############################
# 4) 基本自检（出问题时看日志就能定位）
############################
echo "===== JOB ENV CHECK ====="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node : ${SLURM_NODELIST}"
echo "CWD  : $(pwd)"
echo "PYTHON: ${PYTHON}"
echo "DEMO_DIR: ${DEMO_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "========================="

if [ ! -x "${PYTHON}" ]; then
  echo "[FATAL] Python not found/executable at: ${PYTHON}"
  exit 2
fi

${PYTHON} -V
${PYTHON} -c "import sys; print('sys.executable=', sys.executable)"

echo "----- nvidia-smi -----"
nvidia-smi || true
echo "----------------------"

############################
# 5) 启动训练（使用 srun 更规范）
############################
cd "${DEMO_DIR}"

# 如果你想先跑短调试，把下一行改成：
# srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} ${PYTHON} run_demo.py --stage1a --num-images 5
srun -n 1 -c ${SLURM_CPUS_PER_TASK:-8} \
  ${PYTHON} run_demo.py --stage1a