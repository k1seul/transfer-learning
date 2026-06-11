#!/bin/bash
#SBATCH --job-name=uda_ctop
#SBATCH --partition=gpu          # ← 서버의 파티션 이름으로 변경
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090     # ← GPU 제약조건 (서버에 맞게 수정)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%5           # 13 configs, 최대 5개 동시 실행
#SBATCH --output=logs/ctop_%A_%a.out
#SBATCH --error=logs/ctop_%A_%a.err

# ── 환경 설정 ──────────────────────────────────────
# 아래 중 하나를 사용 (서버 환경에 맞게 수정)

# Option A: uv venv
# source .venv/bin/activate

# Option B: conda
# conda activate uda-search

# Option C: module + conda (클러스터 일반)
# module load cuda/12.1 cudnn/8.9
# conda activate uda-search

# ── wandb 인증 (최초 1회만 필요) ──────────────────
# export WANDB_API_KEY="your_key_here"
# 또는 서버에서 미리 `wandb login` 실행

# ── 데이터 경로 (서버 경로로 변경) ───────────────
DATA_ROOT="/path/to/data"          # ← 수정 필요
CUB_PATH="${DATA_ROOT}/CUB_200_2011/images"
PAINTINGS_PATH="${DATA_ROOT}/CUB_200_Paintings"

# ── 로그 디렉토리 생성 ─────────────────────────────
mkdir -p logs checkpoints

echo "=============================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================="

python run_one.py \
  --config configs/ctop_sweep.json \
  --idx    ${SLURM_ARRAY_TASK_ID}
