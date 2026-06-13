#!/bin/bash
#SBATCH --job-name=ptoc_gray
#SBATCH --partition=P2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#
# Grayscale augmentation ablation — PtoC (Paintings → Photos)
#
# Motivation: CUB paintings dataset에 흑백 이미지가 51.5% 포함.
# Source 모델이 shape/texture 위주 학습 → color 정보 mismatch → pseudo label 품질 저하.
# 해결책: target CUB photos에 RandomGrayscale 적용해 source 분포와 맞춤.
#
# 비교 대상 (이미 완료):
#   [12] shot_only_light  →  best=13.23%  (tgt_gray_p=0.0, baseline)
#
# 이번 실험:
#   [19] shot_gray_tgt04  →  tgt_gray_p=0.4  (shot_only, target만)
#   [20] shot_gray_tgt05  →  tgt_gray_p=0.5  (shot_only, paintings 비율과 동일)
#   [21] shot_gray_both   →  tgt_gray_p=0.5 + src_gray_p=0.4  (양쪽 컬러 불변)
#   [22] jp_gray_tgt05    →  tgt_gray_p=0.5 + joint pseudo
#   [23] jp_gray_best     →  tgt_gray_p=0.5 + joint pseudo + strong SHOT
#
#SBATCH --array=19-23%5
#SBATCH --output=logs/ptoc_gray_%A_%a.out
#SBATCH --error=logs/ptoc_gray_%A_%a.err

source .venv/bin/activate

mkdir -p logs checkpoints

echo "=============================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================="

python run_one.py \
  --config configs/ptoc_sweep.json \
  --idx    ${SLURM_ARRAY_TASK_ID}
