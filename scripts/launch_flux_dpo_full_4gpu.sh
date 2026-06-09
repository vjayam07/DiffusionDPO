#!/bin/bash
#
# NERSC configuration:
##SBATCH --account=m5319
##SBATCH --job-name=dpo-flux-full
##SBATCH --constraint=gpu&hbm80g
##SBATCH --qos=regular
##SBATCH --time=72:00:00
##SBATCH --nodes=1
##SBATCH --gpus=a100:4
##SBATCH --cpus-per-task=16
##SBATCH --ntasks=1
##SBATCH --output=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_full_%j.out
##SBATCH --error=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_full_%j.err
##SBATCH --reservation=cfgrl_experiments_2
#
# Atlas configuration:
#SBATCH --account=atlas
#SBATCH --job-name=dpo-flux-full
#SBATCH --partition=atlas
#SBATCH --gres=gpu:a6000ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=290G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/flux_full_%j.out
#SBATCH --error=/atlas2/u/vjayam/experiments/cfgrl/logs/slurm/flux_full_%j.err

# Launch script for Online DiffusionDPO on FLUX.1-dev with full-weight finetuning (4 GPUs)

# === Paths (shared with DanceGRPO baseline) ===
DATA_DIR="${DATA_DIR:-/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO/data}"
HPS_CKPT_DIR="${HPS_CKPT_DIR:-/atlas2/u/vjayam/experiments/cfgrl-expo/DanceGRPO/hps_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/atlas2/u/vjayam/experiments/cfgrl-expo/DiffusionDPO/output_dpo_flux_full}"

# === Create output directory ===
mkdir -p "${OUTPUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================="
echo " Online DiffusionDPO for FLUX.1-dev (Full FT)"
echo "============================================="
echo "FLUX weights:  ${DATA_DIR}/flux"
echo "Embeddings:    ${DATA_DIR}/rl_embeddings/videos2caption.json"
echo "HPSv2 ckpt:    ${HPS_CKPT_DIR}"
echo "Output:        ${OUTPUT_DIR}"
echo "============================================="

# === Launch training ===
torchrun --nproc_per_node=4 --master_port 19003 \
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
