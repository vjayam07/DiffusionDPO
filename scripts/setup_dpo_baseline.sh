#!/bin/bash
# =============================================================================
# DiffusionDPO Baseline Run — Full Setup for 4x GPU (A100/A6000)
# =============================================================================
# End-to-end guide for running an online DiffusionDPO FLUX LoRA baseline to
# compare against cfgrl-expo / DanceGRPO.
#
# Comparison setup:
#   - Model:      FLUX.1-dev (same as cfgrl-expo / DanceGRPO)
#   - Reward:     HPSv2      (same as cfgrl-expo / DanceGRPO)
#   - Prompts:    HPDv2      (same source, reuses DanceGRPO embeddings)
#   - Resolution: 512x512    (matches all baselines)
#   - Method:     Online DPO with LoRA (rank 128) + adapter toggle
#   - Hardware:   4x GPU (A100 80GB or A6000 48GB)
#
# Quick start:
#   conda create -n dpo-flux python=3.10 -y && conda activate dpo-flux
#   source scripts/setup_dpo_baseline.sh
#   step1_environment
#   step2_checkpoints    # verify data exists
#   step3_embeddings     # verify embeddings exist
#   step4_train          # launch training
#   step5_evaluate       # check results
#
# Or run all at once (after step2 is verified):
#   source scripts/setup_dpo_baseline.sh
#   step1_environment && step3_embeddings && step4_train
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # cd to DiffusionDPO root

DANCEGRPO_DIR="/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO"
DIFFUSIONDPO_DIR="/atlas2/u/vjayam/experiments/cfgrl-expo/DiffusionDPO"
REPO_DIR="$(pwd)"

DATA_DIR="${DANCEGRPO_DIR}/data"
HPS_CKPT_DIR="${DANCEGRPO_DIR}/hps_ckpt"
OUTPUT_DIR="${DIFFUSIONDPO_DIR}/output_dpo_flux"

echo "============================================"
echo " DiffusionDPO Baseline Setup"
echo " Repo:    ${REPO_DIR}"
echo " Data:    ${DATA_DIR}"
echo " Output:  ${OUTPUT_DIR}"
echo "============================================"

# =========================================================
# STEP 1: Environment Setup
# =========================================================
step1_environment() {
    echo ""
    echo ">>> STEP 1: Environment Setup"
    echo "    Python 3.10, PyTorch 2.5, Flash Attention 2, diffusers, peft"
    echo ""

    # Core: PyTorch 2.5 with CUDA 12.1
    pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

    # Flash Attention 2 (compiles for your GPU arch)
    pip install packaging ninja
    pip install flash-attn==2.7.0.post2 --no-build-isolation

    # FLUX-specific deps (newer than the SD/SDXL requirements.txt)
    pip install -r requirements_flux.txt

    # Verify
    python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
import diffusers, peft, wandb
print(f'diffusers {diffusers.__version__}, peft {peft.__version__}')
print('Environment OK')
"

    echo ">>> STEP 1 complete."
}

# =========================================================
# STEP 2: Verify Checkpoints
# =========================================================
step2_checkpoints() {
    echo ""
    echo ">>> STEP 2: Verify Checkpoints"
    echo "    Reuses DanceGRPO's FLUX.1-dev + HPSv2 weights (no re-download)"
    echo ""

    local ok=true

    # FLUX weights
    if [ -d "${DATA_DIR}/flux/transformer" ]; then
        echo "  [OK] FLUX transformer: ${DATA_DIR}/flux/transformer/"
    else
        echo "  [MISSING] FLUX transformer not found at ${DATA_DIR}/flux/transformer/"
        echo "  Run: huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ${DATA_DIR}/flux"
        ok=false
    fi

    if [ -d "${DATA_DIR}/flux/vae" ]; then
        echo "  [OK] FLUX VAE: ${DATA_DIR}/flux/vae/"
    else
        echo "  [MISSING] FLUX VAE not found at ${DATA_DIR}/flux/vae/"
        ok=false
    fi

    # HPSv2 weights
    if [ -f "${HPS_CKPT_DIR}/HPS_v2_compressed.pt" ]; then
        echo "  [OK] HPSv2: ${HPS_CKPT_DIR}/HPS_v2_compressed.pt"
    else
        echo "  [MISSING] HPSv2 not found at ${HPS_CKPT_DIR}/HPS_v2_compressed.pt"
        echo "  Run: huggingface-cli download xswu/HPSv2 HPS_v2_compressed.pt --local-dir ${HPS_CKPT_DIR}"
        ok=false
    fi

    if [ "$ok" = false ]; then
        echo ""
        echo "  Some checkpoints are missing. Download them before proceeding."
        return 1
    fi

    echo ">>> STEP 2 complete. All checkpoints found."
}

# =========================================================
# STEP 3: Verify Embeddings
# =========================================================
step3_embeddings() {
    echo ""
    echo ">>> STEP 3: Verify Preprocessed Embeddings"
    echo "    Reuses DanceGRPO's preprocessed HPDv2 text embeddings"
    echo ""

    local EMBEDDINGS_JSON="${DATA_DIR}/rl_embeddings/videos2caption.json"
    if [ -f "${EMBEDDINGS_JSON}" ]; then
        local NUM_ENTRIES
        NUM_ENTRIES=$(python3 -c "import json; print(len(json.load(open('${EMBEDDINGS_JSON}'))))")
        echo "  [OK] Embeddings: ${NUM_ENTRIES} entries at ${EMBEDDINGS_JSON}"
    else
        echo "  [MISSING] Embeddings not found at ${EMBEDDINGS_JSON}"
        echo "  You need to run DanceGRPO's preprocessing first:"
        echo "    cd ${DANCEGRPO_DIR}"
        echo "    bash scripts/preprocess/preprocess_flux_rl_embeddings_4gpus.sh"
        return 1
    fi

    echo ">>> STEP 3 complete."
}

# =========================================================
# STEP 4: Run Training
# =========================================================
step4_train() {
    echo ""
    echo ">>> STEP 4: Launch DiffusionDPO Training"
    echo "    4x GPU, FLUX LoRA, 512x512, HPSv2 reward, wandb logging"
    echo "    Output: ${OUTPUT_DIR}"
    echo ""

    mkdir -p "${OUTPUT_DIR}"
    cd "${REPO_DIR}"

    export DATA_DIR="${DATA_DIR}"
    export HPS_CKPT_DIR="${HPS_CKPT_DIR}"
    export OUTPUT_DIR="${OUTPUT_DIR}"

    bash scripts/launch_flux_dpo_4gpu.sh

    echo ">>> STEP 4 complete."
}

# =========================================================
# STEP 5: Evaluate Results
# =========================================================
step5_evaluate() {
    echo ""
    echo ">>> STEP 5: Evaluate Results"
    echo ""

    # Reward curve
    local REWARD_FILE="${OUTPUT_DIR}/reward.txt"
    if [ -f "${REWARD_FILE}" ]; then
        echo "--- Reward curve ---"
        echo "File: ${REWARD_FILE}"
        python3 -c "
import numpy as np
data = np.loadtxt('${REWARD_FILE}')
steps, rewards = data[:, 0], data[:, 1]
losses = data[:, 2]
accs = data[:, 3]
print(f'  Steps:        {int(steps[0])} -> {int(steps[-1])}')
print(f'  Reward:       {rewards[0]:.4f} -> {rewards[-1]:.4f} (delta: {rewards[-1]-rewards[0]:+.4f})')
print(f'  Mean reward:  {rewards.mean():.4f}')
print(f'  DPO loss:     {losses[0]:.4f} -> {losses[-1]:.4f}')
print(f'  Implicit acc: {accs[0]:.4f} -> {accs[-1]:.4f}')
"
    else
        echo "  WARNING: No reward.txt found. Training may not have completed."
    fi

    # Eval images
    echo ""
    echo "--- Eval images ---"
    if [ -d "${OUTPUT_DIR}/eval_images" ]; then
        echo "  Saved to: ${OUTPUT_DIR}/eval_images/"
        ls -d "${OUTPUT_DIR}/eval_images/step_"* 2>/dev/null | while read d; do
            echo "    $(basename "$d"): $(ls "$d"/*.png 2>/dev/null | wc -l) images"
        done
    else
        echo "  No eval images found."
    fi

    # Wandb
    echo ""
    echo "--- WandB ---"
    echo "  Check your wandb dashboard (project: flux-dpo)"

    # Checkpoints
    echo ""
    echo "--- Checkpoints ---"
    ls -d "${OUTPUT_DIR}/checkpoint-"* 2>/dev/null | while read d; do
        echo "  $(basename "$d")"
    done

    # Comparison
    echo ""
    echo "--- Comparison with DanceGRPO / cfgrl-expo ---"
    echo "  Key metrics to compare:"
    echo "    - HPSv2 reward curves (reward.txt)"
    echo "    - Wall-clock training time"
    echo "    - Visual quality (wandb eval images)"
    echo "    - DPO implicit accuracy (should be >0.5)"
}

# =========================================================
# Main
# =========================================================
echo ""
echo "Available steps:"
echo "  1) step1_environment   — Create conda env + install deps"
echo "  2) step2_checkpoints   — Verify FLUX + HPSv2 weights"
echo "  3) step3_embeddings    — Verify preprocessed embeddings"
echo "  4) step4_train         — Launch DPO training (4 GPUs)"
echo "  5) step5_evaluate      — Evaluate and compare results"
echo ""
echo "Quick start:"
echo "  conda create -n dpo-flux python=3.10 -y && conda activate dpo-flux"
echo "  source scripts/setup_dpo_baseline.sh"
echo "  step1_environment && step2_checkpoints && step3_embeddings && step4_train"
echo ""
