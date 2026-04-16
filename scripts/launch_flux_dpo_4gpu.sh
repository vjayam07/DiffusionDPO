#!/bin/bash
#
#SBATCH --account=m5319
#SBATCH --job-name=dpo-flux-finetune
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=72:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=16       # CPUs for the job
#SBATCH --ntasks=4            # Number of tasks (one per GPU)
#SBATCH --output=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_finetune_%j.out
#SBATCH --error=/pscratch/sd/v/vjayam/DiffusionDPO/slurm_logs/flux_finetune_%j.err
#SBATCH --reservation=cfgrl_expo

# Launch script for Online DiffusionDPO on FLUX.1-dev with LoRA (4 GPUs)
# Mirrors DanceGRPO's finetune_flux_grpo_4gpus_lora_a6000.sh

# === Paths (shared with DanceGRPO baseline) ===
DATA_DIR="${DATA_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/data}"
HPS_CKPT_DIR="${HPS_CKPT_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/hps_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/output_dpo_flux}"
EVAL_PROMPTS_DIR="${EVAL_PROMPTS_DIR:-/pscratch/sd/v/vjayam/DiffusionDPO/embeddings/flux_hpdv2}"

# === Create output directory ===
mkdir -p "${OUTPUT_DIR}"

echo "============================================="
echo " Online DiffusionDPO for FLUX.1-dev (LoRA)"
echo "============================================="
echo "FLUX weights:  ${DATA_DIR}/flux"
echo "Embeddings:    ${DATA_DIR}/rl_embeddings/videos2caption.json"
echo "HPSv2 ckpt:    ${HPS_CKPT_DIR}"
echo "Eval prompts:  ${EVAL_PROMPTS_DIR}"
echo "Output:        ${OUTPUT_DIR}"
echo "============================================="

# === Launch training ===
torchrun --nproc_per_node=4 --master_port 19003 \
  train_flux_dpo_lora.py \
  --pretrained_model_name_or_path "${DATA_DIR}/flux" \
  --data_json_path "${DATA_DIR}/rl_embeddings/videos2caption.json" \
  --h 512 --w 512 \
  --sampling_steps 8 \
  --shift 3.0 \
  --guidance 3.5 \
  --lora_rank 128 \
  --lora_alpha 256 \
  --beta_dpo 5000 \
  --num_generations 4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 12 \
  --learning_rate 3e-4 \
  --max_train_steps 1000 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --hps_ckpt_dir "${HPS_CKPT_DIR}" \
  --seed 42 \
  --checkpointing_steps 100 \
  --log_every 1 \
  --max_grad_norm 1.0 \
  --wandb_project "flux-dpo" \
  --max_eval_images 9 \
  --eval_prompts_dir "${EVAL_PROMPTS_DIR}" \
  --cfgrl_eval_seed 0 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
