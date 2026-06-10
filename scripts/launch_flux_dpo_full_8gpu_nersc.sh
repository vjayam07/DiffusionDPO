#!/bin/bash
#
# Submit from the repository root with:
#   sbatch scripts/launch_flux_dpo_full_8gpu_nersc.sh
#
# Override paths or training settings at submission time, for example:
#   DATA_DIR=/pscratch/sd/v/vjayam/DiffusionDPO/data \
#   OUTPUT_DIR=/pscratch/sd/v/vjayam/DiffusionDPO/output_dpo_flux_full \
#   sbatch scripts/launch_flux_dpo_full_8gpu_nersc.sh
#
#SBATCH --account=m5319
#SBATCH --job-name=dpo-flux-full
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_full_8gpu_nersc_%j.out
#SBATCH --error=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_full_8gpu_nersc_%j.err
#SBATCH --reservation=final_runs

set -euo pipefail

DATA_DIR="${DATA_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/data}"
HPS_CKPT_DIR="${HPS_CKPT_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/hps_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/output_dpo_flux_full}"

# Run from the directory where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set; submit this script with sbatch}"

if [[ ! -f "train_flux_dpo_full.py" ]]; then
  echo "train_flux_dpo_full.py not found in submission directory: ${SLURM_SUBMIT_DIR}" >&2
  echo "Run sbatch from the DiffusionDPO repository root." >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}/flux/transformer" || ! -d "${DATA_DIR}/flux/vae" ]]; then
  echo "FLUX.1-dev weights not found under: ${DATA_DIR}/flux" >&2
  echo "Set DATA_DIR to the DanceGRPO-compatible data directory." >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/rl_embeddings/videos2caption.json" ]]; then
  echo "Baseline prompt embeddings not found: ${DATA_DIR}/rl_embeddings/videos2caption.json" >&2
  echo "Set DATA_DIR to the DanceGRPO-compatible data directory." >&2
  exit 1
fi
if [[ ! -f "${HPS_CKPT_DIR}/HPS_v2_compressed.pt" ]]; then
  echo "HPSv2 checkpoint not found: ${HPS_CKPT_DIR}/HPS_v2_compressed.pt" >&2
  echo "Set HPS_CKPT_DIR to the DanceGRPO-compatible HPS checkpoint directory." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_ASYNC_ERROR_HANDLING=1

# Avoid collisions if multiple jobs share a node during interactive testing.
MASTER_PORT="${MASTER_PORT:-$((19000 + SLURM_JOB_ID % 1000))}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')}"

echo "============================================="
echo " Online DiffusionDPO for FLUX.1-dev (Full FT)"
echo " NERSC Perlmutter: 8x A100 80GB across 2 nodes"
echo "============================================="
echo "Repository:    ${SLURM_SUBMIT_DIR}"
echo "FLUX weights:  ${DATA_DIR}/flux"
echo "Embeddings:    ${DATA_DIR}/rl_embeddings/videos2caption.json"
echo "HPSv2 ckpt:    ${HPS_CKPT_DIR}"
echo "Output:        ${OUTPUT_DIR}"
echo "Master addr:   ${MASTER_ADDR}"
echo "Master port:   ${MASTER_PORT}"
echo "============================================="

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores --gpu-bind=none \
  torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node=4 \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
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
  --gradient_accumulation_steps 6 \
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
