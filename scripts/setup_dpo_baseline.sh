#!/bin/bash
# End-to-end setup for DiffusionDPO baseline (FLUX.1-dev)
# Mirrors DanceGRPO's setup_baseline_run.sh
#
# 5 steps:
#   1. Environment (torch 2.5, flash-attn, diffusers, peft, hpsv2)
#   2. Checkpoints (reuse DanceGRPO's FLUX weights + HPSv2 weights)
#   3. Embeddings (reuse DanceGRPO's preprocessed HPDv2 embeddings)
#   4. Training (launch train_flux_dpo_lora.py)
#   5. Evaluation (compare reward curves)

set -euo pipefail

# === Configuration ===
DANCEGRPO_DIR="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"
DIFFUSIONDPO_DIR="/atlas2/u/vjayam/experiments/cfgrl-expo/DiffusionDPO"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DANCEGRPO_DIR}/data"
HPS_CKPT_DIR="${DANCEGRPO_DIR}/hps_ckpt"
OUTPUT_DIR="${DIFFUSIONDPO_DIR}/output_dpo_flux"

echo "============================================="
echo " DiffusionDPO Baseline Setup"
echo "============================================="
echo "Repo:          ${REPO_DIR}"
echo "DanceGRPO:     ${DANCEGRPO_DIR}"
echo "Output:        ${OUTPUT_DIR}"
echo "============================================="

# =============================================
# Step 1: Environment
# =============================================
echo ""
echo ">>> Step 1/5: Environment setup"

# Check Python and CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Install dependencies (if not already present)
pip install --quiet diffusers>=0.32.0 peft transformers accelerate
pip install --quiet hpsv2

echo "  Environment OK"

# =============================================
# Step 2: Checkpoints
# =============================================
echo ""
echo ">>> Step 2/5: Verify checkpoints"

# FLUX weights (reuse from DanceGRPO)
if [ -d "${DATA_DIR}/flux" ]; then
    echo "  FLUX weights found at ${DATA_DIR}/flux"
else
    echo "  ERROR: FLUX weights not found at ${DATA_DIR}/flux"
    echo "  Please ensure DanceGRPO data is available."
    exit 1
fi

# HPSv2 weights (reuse from DanceGRPO)
if [ -f "${HPS_CKPT_DIR}/HPS_v2_compressed.pt" ]; then
    echo "  HPSv2 weights found at ${HPS_CKPT_DIR}/HPS_v2_compressed.pt"
else
    echo "  ERROR: HPSv2 weights not found at ${HPS_CKPT_DIR}/HPS_v2_compressed.pt"
    echo "  Please ensure DanceGRPO HPSv2 checkpoint is available."
    exit 1
fi

# =============================================
# Step 3: Embeddings
# =============================================
echo ""
echo ">>> Step 3/5: Verify embeddings"

EMBEDDINGS_JSON="${DATA_DIR}/rl_embeddings/videos2caption.json"
if [ -f "${EMBEDDINGS_JSON}" ]; then
    NUM_ENTRIES=$(python3 -c "import json; print(len(json.load(open('${EMBEDDINGS_JSON}'))))")
    echo "  Embeddings found: ${NUM_ENTRIES} entries at ${EMBEDDINGS_JSON}"
else
    echo "  ERROR: Embeddings not found at ${EMBEDDINGS_JSON}"
    echo "  Please ensure DanceGRPO preprocessed embeddings are available."
    exit 1
fi

# =============================================
# Step 4: Training
# =============================================
echo ""
echo ">>> Step 4/5: Launch training"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"
export DATA_DIR="${DATA_DIR}"
export HPS_CKPT_DIR="${HPS_CKPT_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR}"

bash scripts/launch_flux_dpo_4gpu.sh

# =============================================
# Step 5: Evaluation
# =============================================
echo ""
echo ">>> Step 5/5: Evaluation summary"

REWARD_FILE="${OUTPUT_DIR}/reward.txt"
if [ -f "${REWARD_FILE}" ]; then
    echo "  Reward log: ${REWARD_FILE}"
    echo "  First 5 entries:"
    head -5 "${REWARD_FILE}" | sed 's/^/    /'
    echo "  Last 5 entries:"
    tail -5 "${REWARD_FILE}" | sed 's/^/    /'

    # Quick stats
    python3 -c "
import numpy as np
data = np.loadtxt('${REWARD_FILE}')
steps, rewards = data[:, 0], data[:, 1]
print(f'  Steps: {int(steps[0])} -> {int(steps[-1])}')
print(f'  Reward: {rewards[0]:.4f} -> {rewards[-1]:.4f} (delta: {rewards[-1]-rewards[0]:+.4f})')
print(f'  Mean reward: {rewards.mean():.4f}')
"
else
    echo "  WARNING: No reward.txt found. Training may not have completed."
fi

echo ""
echo "============================================="
echo " DiffusionDPO baseline complete!"
echo " Output: ${OUTPUT_DIR}"
echo " Compare reward curves with DanceGRPO and cfgrl-expo baselines."
echo "============================================="
