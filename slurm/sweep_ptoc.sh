#!/bin/bash
#SBATCH --job-name=uda_ptoc
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=19-23%5          # grayscale aug: shot_gray_tgt04/05, shot_gray_both, jp_gray_tgt05, jp_gray_best
#SBATCH --output=logs/ptoc_%A_%a.out
#SBATCH --error=logs/ptoc_%A_%a.err

# ── 환경 설정 (sweep_ctop.sh와 동일하게 수정) ──────
source .venv/bin/activate

DATA_ROOT="/path/to/data"
CUB_PATH="${DATA_ROOT}/CUB_200_2011/images"
PAINTINGS_PATH="${DATA_ROOT}/CUB_200_Paintings"

mkdir -p logs checkpoints

echo "=============================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================="

python run_one.py \
  --config configs/ptoc_sweep.json \
  --idx    ${SLURM_ARRAY_TASK_ID}
