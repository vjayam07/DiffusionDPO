#!/bin/bash
#
# Submit from the repository root with:
#   sbatch scripts/launch_flux_dpo_full_4gpu_nersc.sh
#
# Override paths or training settings at submission time, for example:
#   DATA_DIR=/pscratch/sd/v/vjayam/DiffusionDPO/data \
#   OUTPUT_DIR=/pscratch/sd/v/vjayam/DiffusionDPO/output_dpo_flux_full \
#   sbatch scripts/launch_flux_dpo_full_4gpu_nersc.sh
#
#SBATCH --account=m5319
#SBATCH --job-name=dpo-flux-full
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_full_nersc_%j.out
#SBATCH --error=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_full_nersc_%j.err

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATA_DIR="${DATA_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/data}"
HPS_CKPT_DIR="${HPS_CKPT_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/hps_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/output_dpo_flux_full}"

mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING=1

# Avoid collisions if multiple jobs share a node during interactive testing.
MASTER_PORT="${MASTER_PORT:-$((19000 + SLURM_JOB_ID % 1000))}"

echo "============================================="
echo " Online DiffusionDPO for FLUX.1-dev (Full FT)"
echo " NERSC Perlmutter: 4x A100 80GB"
echo "============================================="
echo "Repository:    ${REPO_DIR}"
echo "FLUX weights:  ${DATA_DIR}/flux"
echo "Embeddings:    ${DATA_DIR}/rl_embeddings/videos2caption.json"
echo "HPSv2 ckpt:    ${HPS_CKPT_DIR}"
echo "Output:        ${OUTPUT_DIR}"
echo "Master port:   ${MASTER_PORT}"
echo "============================================="

srun --ntasks=1 --cpu-bind=cores \
  torchrun --nproc_per_node=4 --master_port "${MASTER_PORT}" \
  train_flux_dpo_full.py \
  --pretrained_model_name_or_path "${DATA_DIR}/flux" \
  --data_json_path "${DATA_DIR}/rl_embeddings/videos2caption.json" \
  --h 512 --w 512 \
  --sampling_steps 4 \
  --shift 3.0 \
  --guidance 3.5 \
  --beta_dpo 5000 \
  --num_generations 2 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 12 \
  --learning_rate 1e-6 \
  --max_train_steps 4500 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --hps_ckpt_dir "${HPS_CKPT_DIR}" \
  --seed 42 \
  --checkpointing_steps 100 \
  --log_every 1 \
  --max_grad_norm 1.0 \
  --wandb_project "flux-dpo-full" \
  --max_eval_images 100 \
  --num_train_prompts 75000 \
  --num_eval_prompts 500 \
  --prompt_seed 42 \
  --train_test_split 0.8 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
