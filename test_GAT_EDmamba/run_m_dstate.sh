#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# mamba_ssm.Mamba requires CUDA.
python - <<'PY'
import torch
if not torch.cuda.is_available():
  raise SystemExit("ERROR: CUDA not available, but mamba_ssm.Mamba requires CUDA.\n"
           "Please run on a machine with GPU/CUDA and a CUDA-enabled PyTorch.")
print("CUDA available:", torch.cuda.get_device_name(0))
PY

# Fixed seed plan (as requested).
SEEDS=(33)
# 2 17 27 30 33 51 62 80 88 97
DATASETS=(FD002)
# FD002 FD003 FD004

mamba_states=(32)
MAX_EPOCHS="${MAX_EPOCHS:-30}"
MODEL_CODE="${MODEL_CODE:-GAT_EDmamba_dstate}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.002}"

for DATASET in "${DATASETS[@]}"; do

  for mamba_state in "${mamba_states[@]}"; do
    echo "Training for DATASET=${DATASET}, mamba_state=${mamba_state}"
    if [[ "$DATASET" == "FD001" || "$DATASET" == "FD003" ]]; then
      SMOOTH_RATE=30
    else
      SMOOTH_RATE=40
    fi  
    if [ "$DATASET" == "FD001" ]; then
      mamba_d_model=4
      alpha=1.5
      gamma=2.0
      delta=0.9
      focus_threshold=35.0
      cap_threshold=125.0
    elif [ "$DATASET" == "FD002" ]; then
      mamba_d_model=16
      alpha=3.0
      gamma=5.0
      delta=0.9
      focus_threshold=35.0
      cap_threshold=125.0
    elif [ "$DATASET" == "FD003" ]; then
      mamba_d_model=32
      alpha=3.5
      gamma=5.0
      delta=0.9
      focus_threshold=35.0
      cap_threshold=125.0
    elif [ "$DATASET" == "FD004" ]; then
      mamba_d_model=8
      alpha=2.0
      gamma=5.0
      delta=0.9
      focus_threshold=35.0
      cap_threshold=125.0
    fi

    echo "====================================="
    echo " 训练开始 | DATASET=${DATASET} | smooth_rate=${SMOOTH_RATE} | max_epochs=${MAX_EPOCHS}"
    echo "Seeds: ${SEEDS[*]}"
    echo "====================================="

    PYTHONPATH="$PROJECT_ROOT" python test_GAT_EDmamba/train_model.py \
      --sub-dataset "$DATASET" \
      --max-epochs "$MAX_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --smooth-rate "$SMOOTH_RATE" \
      --seed-list "${SEEDS[@]}" \
      --model-code "$MODEL_CODE"_"$mamba_state" \
      --asym-alpha "$alpha" \
      --asym-gamma "$gamma" \
      --asym-delta "$delta" \
      --focus-threshold "$focus_threshold" \
      --cap-threshold "$cap_threshold" \
      --mamba-d-state "$mamba_state" \
      --mamba-d-model "$mamba_d_model"
  done
done

echo "全部数据集已执行完成: ${DATASETS[*]}"