#!/bin/bash
#SBATCH --job-name=uda_multiseed
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/multiseed_%A_%a.out
#SBATCH --error=logs/multiseed_%A_%a.err

source .venv/bin/activate

mkdir -p logs checkpoints

echo "=============================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================="

python run_one.py \
  --config configs/best_multiseed.json \
  --idx    ${SLURM_ARRAY_TASK_ID}
