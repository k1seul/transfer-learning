#!/bin/bash
#SBATCH --job-name=ssl_ptoc
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/ssl_ptoc_%A_%a.out
#SBATCH --error=logs/ssl_ptoc_%A_%a.err

source .venv/bin/activate

mkdir -p logs checkpoints

echo "=============================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================="

python run_one.py \
  --config configs/ssl_ptoc.json \
  --idx    ${SLURM_ARRAY_TASK_ID}
