#!/usr/bin/env python
"""
Online DiffusionDPO for FLUX.1-dev with LoRA + HPSv2 AI feedback.

Generates multiple images per prompt, scores with HPSv2, selects best/worst
as preference pair, applies DPO loss adapted for FLUX flow matching.
Reference model implemented via LoRA adapter toggle (no extra memory).

Usage:
    torchrun --nproc_per_node=4 train_flux_dpo_lora.py --pretrained_model_name_or_path ...
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FLUX helpers (flow-matching utilities)
# ---------------------------------------------------------------------------

def sd3_time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Shift the time schedule for SD3/FLUX (from SD3 paper)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def pack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Pack spatial latents into sequence for FLUX transformer.
    Input: (B, C, H, W) -> Output: (B, H*W/4, C*4)
    """
    b, c, lh, lw = latents.shape
    latents = latents.reshape(b, c, lh // 2, 2, lw // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, lh//2, lw//2, C, 2, 2)
    latents = latents.reshape(b, (lh // 2) * (lw // 2), c * 4)
    return latents


def unpack_latents(latents: torch.Tensor, h: int, w: int, vae_scale_factor: int = 8) -> torch.Tensor:
    """Unpack sequence back to spatial latents.
    Input: (B, seq, C*4) -> Output: (B, C, H, W)
    """
    lh = h // vae_scale_factor
    lw = w // vae_scale_factor
    b, seq, dim = latents.shape
    c = dim // 4
    latents = latents.reshape(b, lh // 2, lw // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, lh//2, 2, lw//2, 2)
    latents = latents.reshape(b, c, lh, lw)
    return latents


def prepare_latent_image_ids(batch_size: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create image position IDs for FLUX transformer."""
    lh = h // 2  # after packing
    lw = w // 2
    ids = torch.zeros(lh, lw, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(lh, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(lw, device=device, dtype=dtype)[None, :]
    ids = ids.reshape(lh * lw, 3)
    ids = ids.unsqueeze(0).expand(batch_size, -1, -1)
    return ids


# ---------------------------------------------------------------------------
# Dataset: loads precomputed FLUX text embeddings
# ---------------------------------------------------------------------------

class LatentDataset(Dataset):
    """Dataset that loads precomputed text embeddings for FLUX RL training.

    Matches DanceGRPO's latent_flux_rl_datasets.py format:
    - JSON contains bare filenames (e.g. "36044.pt")
    - Actual .pt files live in subdirectories next to the JSON:
        <json_dir>/prompt_embed/<filename>.pt
        <json_dir>/pooled_prompt_embeds/<filename>.pt
        <json_dir>/text_ids/<filename>.pt
    """

    def __init__(self, data_json_path: str):
        with open(data_json_path, "r") as f:
            self.data = json.load(f)
        # Base directories derived from JSON location (matches DanceGRPO)
        base_dir = os.path.dirname(data_json_path)
        self.prompt_embed_dir = os.path.join(base_dir, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(base_dir, "pooled_prompt_embeds")
        self.text_ids_dir = os.path.join(base_dir, "text_ids")
        logger.info(f"Loaded {len(self.data)} entries from {data_json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt_embeds = torch.load(
            os.path.join(self.prompt_embed_dir, entry["prompt_embed_path"]),
            map_location="cpu", weights_only=True,
        )
        pooled_prompt_embeds = torch.load(
            os.path.join(self.pooled_prompt_embeds_dir, entry["pooled_prompt_embeds_path"]),
            map_location="cpu", weights_only=True,
        )
        text_ids = torch.load(
            os.path.join(self.text_ids_dir, entry["text_ids"]),
            map_location="cpu", weights_only=True,
        )
        # text_ids is stored as a template (1, 3); expand to match prompt
        # embedding sequence length (DanceGRPO does text_ids.repeat(seq_len, 1))
        text_seq_len = prompt_embeds.shape[0]
        if text_ids.shape[0] < text_seq_len:
            text_ids = text_ids.expand(text_seq_len, -1).contiguous()
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
            "prompt": entry.get("caption", ""),
        }


def collate_fn(batch):
    return {
        "prompt_embeds": torch.stack([b["prompt_embeds"] for b in batch]),
        "pooled_prompt_embeds": torch.stack([b["pooled_prompt_embeds"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
    }


# ---------------------------------------------------------------------------
# HPSv2 reward scoring
# ---------------------------------------------------------------------------

def load_hps_model(hps_ckpt_dir: str, device: torch.device):
    """Load HPSv2 reward model."""
    from utils.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14", "laion2B-s32B-b79K",
        precision="amp", device=device, jit=False,
        force_quick_gelu=False, force_custom_text=False,
        force_patch_dropout=False, force_image_size=None,
        pretrained_image=False, image_mean=None, image_std=None,
        light_augmentation=True, aug_cfg={}, output_dict=True,
        with_score_predictor=False, with_region_predictor=False,
    )
    ckpt_path = os.path.join(hps_ckpt_dir, "HPS_v2_compressed.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    model = model.to(device).eval()
    return model, preprocess_val, tokenizer


@torch.no_grad()
def score_images(hps_model, hps_preprocess, hps_tokenizer, images, prompts, device):
    """Score a list of PIL images with HPSv2. Returns list of float scores."""
    scores = []
    for img, prompt in zip(images, prompts):
        image_tensor = hps_preprocess(img).unsqueeze(0).to(device, non_blocking=True)
        text = hps_tokenizer([prompt]).to(device, non_blocking=True)
        outputs = hps_model(image_tensor, text)
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        score = (image_features @ text_features.T).diagonal().item()
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# FLUX model loading with LoRA
# ---------------------------------------------------------------------------

def load_flux_models(args, device):
    """Load FLUX transformer, VAE, and scheduler; apply LoRA."""
    import gc
    from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
    from peft import LoraConfig, get_peft_model

    logger.info(f"Loading FLUX from {args.pretrained_model_name_or_path}")
    pipe = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # Free text encoders — not needed since we use precomputed embeddings.
    # T5-XXL alone is ~22GB in bf16.
    del pipe.text_encoder, pipe.text_encoder_2
    del pipe
    gc.collect()

    # Freeze everything
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # Apply LoRA to transformer
    # Workaround: PEFT's cast_adapter_dtype enumerates float8 dtypes
    # (e.g. float8_e8m0fnu) that don't exist in PyTorch 2.5. Patch it
    # to silently skip missing dtypes.
    import peft.tuners.tuners_utils as _peft_tuners
    _orig_cast = _peft_tuners.cast_adapter_dtype
    def _safe_cast(*args, **kwargs):
        try:
            return _orig_cast(*args, **kwargs)
        except AttributeError:
            pass
    _peft_tuners.cast_adapter_dtype = _safe_cast

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    vae = vae.to(device)
    vae.eval()

    return transformer, vae, scheduler


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def setup_fsdp(transformer, device, args):
    """Wrap transformer with FSDP for distributed training."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
    import functools

    # Wrap policy: wrap each transformer block
    def _policy_fn(module):
        cls_name = module.__class__.__name__
        return "TransformerBlock" in cls_name or "SingleTransformerBlock" in cls_name

    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=_policy_fn)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    transformer = FSDP(
        transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )
    return transformer


# ---------------------------------------------------------------------------
# Generation: Euler sampling for FLUX
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_latents(transformer, scheduler, batch, args, device):
    """Generate latent images using Euler sampling (no grad).
    Returns: latents (B*K, C, H, W) in latent space.
    """
    B = batch["prompt_embeds"].shape[0]
    K = args.num_generations
    lh = args.h // 8  # VAE scale factor = 8
    lw = args.w // 8
    C = 16  # FLUX latent channels

    # Repeat prompt embeddings K times
    prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device, dtype=torch.bfloat16)
    text_ids = batch["text_ids"].to(device, dtype=torch.bfloat16)

    prompt_embeds = prompt_embeds.repeat_interleave(K, dim=0)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(K, dim=0)
    text_ids = text_ids.repeat_interleave(K, dim=0)

    BK = B * K
    latent_image_ids = prepare_latent_image_ids(BK, lh, lw, device, torch.bfloat16)

    # Initialize noise
    latents = torch.randn(BK, C, lh, lw, device=device, dtype=torch.bfloat16)

    # Setup timesteps
    sigmas = torch.linspace(1.0, 0.0, args.sampling_steps + 1, device=device)
    # Apply shift
    sigmas = sd3_time_shift(args.shift, 1.0, sigmas)

    # Ensure LoRA is ON during generation (we generate from current policy)
    # PeftModel methods: enable_adapter_layers / disable_adapter_layers
    peft_model = transformer.module if hasattr(transformer, 'module') else transformer
    peft_model.enable_adapter_layers()

    for i in range(args.sampling_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        t = sigma.expand(BK)

        # Pack latents for transformer
        packed = pack_latents(latents, args.h, args.w)

        # Transformer forward — FLUX expects sigma directly in [0, 1]
        timestep = t
        noise_pred = transformer(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            guidance=torch.tensor([args.guidance], device=device, dtype=torch.bfloat16).expand(BK),
            return_dict=False,
        )[0]

        # Unpack
        noise_pred = unpack_latents(noise_pred, args.h, args.w)

        # Euler step: x_{t-1} = x_t + (sigma_next - sigma) * v
        latents = latents + (sigma_next - sigma) * noise_pred

    return latents  # (BK, C, lh, lw)


@torch.no_grad()
def decode_latents(vae, latents):
    """Decode latents to PIL images."""
    # FLUX VAE scaling
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    images = vae.decode(latents.float()).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).cpu().float().numpy()
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    return pil_images


# ---------------------------------------------------------------------------
# DPO training step with adapter toggle
# ---------------------------------------------------------------------------

def dpo_training_step(transformer, vae, batch, latents_w, latents_l, args, device):
    """
    Compute flow-matching DPO loss.

    1. Sample random sigma, create noisy versions of preferred/unpreferred latents
    2. Predict velocity with LoRA ON (policy) and LoRA OFF (reference)
    3. Compute DPO implicit reward loss

    Returns: loss, metrics dict
    """
    B = latents_w.shape[0]
    C, lh, lw = latents_w.shape[1:]

    prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device, dtype=torch.bfloat16)
    text_ids = batch["text_ids"].to(device, dtype=torch.bfloat16)
    latent_image_ids = prepare_latent_image_ids(B, lh, lw, device, torch.bfloat16)

    # Sample random sigma (timestep) for each item in batch
    sigma = torch.rand(B, device=device, dtype=torch.bfloat16)
    sigma = sd3_time_shift(args.shift, 1.0, sigma)
    sigma = sigma[:, None, None, None]  # (B, 1, 1, 1)

    # Sample shared noise
    noise = torch.randn_like(latents_w)

    # Flow matching forward process: x_t = (1 - sigma) * x_0 + sigma * noise
    noisy_w = (1 - sigma) * latents_w + sigma * noise
    noisy_l = (1 - sigma) * latents_l + sigma * noise

    # Target velocity: v = noise - x_0 (FLUX flow matching convention)
    target_w = noise - latents_w
    target_l = noise - latents_l

    # Flatten sigma for timestep input
    t = sigma.squeeze()  # (B,)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    timestep = t  # FLUX expects sigma directly in [0, 1]

    guidance_vec = torch.tensor([args.guidance], device=device, dtype=torch.bfloat16).expand(B)

    # --- Policy prediction (LoRA ON) ---
    peft_model = transformer.module if hasattr(transformer, 'module') else transformer
    peft_model.enable_adapter_layers()

    # Predict for preferred
    packed_w = pack_latents(noisy_w, args.h, args.w)
    model_pred_w = transformer(
        hidden_states=packed_w,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        guidance=guidance_vec,
        return_dict=False,
    )[0]
    model_pred_w = unpack_latents(model_pred_w, args.h, args.w)

    # Predict for unpreferred
    packed_l = pack_latents(noisy_l, args.h, args.w)
    model_pred_l = transformer(
        hidden_states=packed_l,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        guidance=guidance_vec,
        return_dict=False,
    )[0]
    model_pred_l = unpack_latents(model_pred_l, args.h, args.w)

    # --- Reference prediction (LoRA OFF) ---
    peft_model.disable_adapter_layers()

    with torch.no_grad():
        ref_pred_w = transformer(
            hidden_states=packed_w,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            guidance=guidance_vec,
            return_dict=False,
        )[0]
        ref_pred_w = unpack_latents(ref_pred_w, args.h, args.w)

        ref_pred_l = transformer(
            hidden_states=packed_l,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            guidance=guidance_vec,
            return_dict=False,
        )[0]
        ref_pred_l = unpack_latents(ref_pred_l, args.h, args.w)

    # Re-enable LoRA for gradient computation
    peft_model.enable_adapter_layers()

    # --- DPO loss (from DiffusionDPO paper, adapted for flow matching) ---
    # Cast to float32 for stable loss computation (matches original DiffusionDPO)
    model_losses_w = (model_pred_w.float() - target_w.float()).pow(2).mean(dim=[1, 2, 3])
    model_losses_l = (model_pred_l.float() - target_l.float()).pow(2).mean(dim=[1, 2, 3])
    model_diff = model_losses_w - model_losses_l

    ref_losses_w = (ref_pred_w.float() - target_w.float()).pow(2).mean(dim=[1, 2, 3])
    ref_losses_l = (ref_pred_l.float() - target_l.float()).pow(2).mean(dim=[1, 2, 3])
    ref_diff = ref_losses_w - ref_losses_l

    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
    raw_ref_loss = 0.5 * (ref_losses_w.mean() + ref_losses_l.mean())

    scale_term = -0.5 * args.beta_dpo
    inside_term = scale_term * (model_diff - ref_diff)
    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    loss = -1 * F.logsigmoid(inside_term).mean()

    metrics = {
        "dpo_loss": loss.item(),
        "implicit_acc": implicit_acc.item(),
        "model_mse": raw_model_loss.item(),
        "ref_mse": raw_ref_loss.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_lora_checkpoint(transformer, optimizer, step, output_dir):
    """Save LoRA adapter weights using FSDP full state dict gathering."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
    )

    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    # Gather full state dict from all FSDP shards onto rank 0 CPU
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = transformer.state_dict()
        # Save only LoRA weights
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}

    if dist.get_rank() == 0:
        torch.save(lora_state_dict, os.path.join(save_dir, "lora_weights.pt"))
        torch.save({"step": step}, os.path.join(save_dir, "training_state.pt"))
        logger.info(f"Saved checkpoint at step {step} to {save_dir}")


# ---------------------------------------------------------------------------
# Compute tracking: calibration
# ---------------------------------------------------------------------------

def calibrate_compute_tracker(tracker, transformer, vae, hps_model, hps_tokenizer,
                              device, args):
    """Measure actual FLOPs for each phase using FlopCounterMode.

    Runs one forward pass of each component with real-shaped inputs to calibrate
    the tracker. Called once before the training loop.
    """
    lh = args.h // 8
    lw = args.w // 8
    C = 16

    # Dummy inputs matching actual shapes
    z = torch.randn(1, C, lh, lw, device=device, dtype=torch.bfloat16)
    z_packed = pack_latents(z, args.h, args.w)
    image_ids = prepare_latent_image_ids(1, lh, lw, device, torch.bfloat16)
    encoder_hidden_states = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)
    pooled_prompt_embeds = torch.zeros(1, 768, device=device, dtype=torch.bfloat16)
    text_ids = torch.zeros(1, 512, 3, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)  # sigma in [0, 1]
    guidance = torch.tensor([3.5], device=device, dtype=torch.bfloat16)

    # 1. Calibrate: one FLUX transformer forward pass (sampling)
    transformer.eval()
    with tracker.calibrate_sampling():
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                transformer(
                    hidden_states=z_packed,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    guidance=guidance,
                    return_dict=False,
                )

    # 2. Calibrate: VAE decode
    latent_for_vae = torch.randn(1, C, lh, lw, device=device, dtype=torch.bfloat16)
    with tracker.calibrate_vae_decode():
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vae.decode(latent_for_vae, return_dict=False)

    # 3. Calibrate: reward model forward
    if hps_model is not None:
        dummy_image = torch.randn(1, 3, 224, 224, device=device)
        dummy_text = hps_tokenizer(["a photo"]).to(device=device)
        with tracker.calibrate_reward():
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    hps_model(dummy_image, dummy_text)

    # 4. Calibrate: one DPO training step (fwd + bwd)
    transformer.train()
    with tracker.calibrate_training():
        with torch.autocast("cuda", torch.bfloat16):
            pred = transformer(
                hidden_states=z_packed,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
                guidance=guidance,
                return_dict=False,
            )[0]
        loss = pred.sum()
        loss.backward()
    # Clear gradients from calibration
    for p in transformer.parameters():
        if p.grad is not None:
            p.grad.zero_()

    tracker.print_calibration_summary()


# ---------------------------------------------------------------------------
# Evaluation: generate fixed-seed images and log to wandb
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_and_log_images(args, transformer, vae, dataset, device, step,
                        eval_seed=42, max_images=9):
    """Generate images with fixed prompts/noise and log to wandb.

    Mirrors cfgrl-expo's evaluation: same seed resets every eval round so that
    the same prompts and initial latents are used, enabling apples-to-apples
    visual comparison across methods.

    All ranks run the transformer forward (required by FSDP), but only rank 0
    decodes and logs images.
    """
    from diffusers.image_processor import VaeImageProcessor

    rank = dist.get_rank()
    transformer.eval()

    lh = args.h // 8
    lw = args.w // 8
    C = 16

    # Reset noise generator every eval round
    generator = torch.Generator(device=device).manual_seed(eval_seed)

    # Sigma schedule (same as training)
    sigmas = torch.linspace(1.0, 0.0, args.sampling_steps + 1, device=device)
    sigmas = sd3_time_shift(args.shift, 1.0, sigmas)

    num_eval = min(max_images, len(dataset))

    # Phase 1: all ranks run transformer forward (required by FSDP)
    all_z_packed = []
    all_captions = []

    # Ensure LoRA is ON (generate from current policy)
    peft_model = transformer.module if hasattr(transformer, 'module') else transformer
    peft_model.enable_adapter_layers()

    for idx in range(num_eval):
        entry = dataset[idx]
        prompt_embeds = entry["prompt_embeds"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        pooled_prompt_embeds = entry["pooled_prompt_embeds"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        text_ids = entry["text_ids"].unsqueeze(0).to(device, dtype=torch.bfloat16)
        caption = entry["prompt"]

        z = torch.randn(1, C, lh, lw, device=device, dtype=torch.bfloat16,
                        generator=generator)
        z_packed = pack_latents(z, args.h, args.w)
        image_ids = prepare_latent_image_ids(1, lh, lw, device, torch.bfloat16)

        # Deterministic Euler sampling
        for i in range(args.sampling_steps):
            sigma = sigmas[i]
            timestep = sigma.unsqueeze(0)  # FLUX expects sigma in [0, 1]
            with torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    hidden_states=z_packed,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    guidance=torch.tensor([args.guidance], device=device,
                                          dtype=torch.bfloat16),
                    return_dict=False,
                )[0]
            dsigma = sigmas[i + 1] - sigmas[i]
            z_packed = z_packed + dsigma * pred

        all_z_packed.append(z_packed)
        all_captions.append(caption)

    # Phase 2: only rank 0 decodes and logs (VAE is not FSDP-wrapped)
    if rank == 0:
        vae.enable_tiling()
        image_processor = VaeImageProcessor(vae_scale_factor=16)
        wandb_images = []
        saved_paths = []
        images_dir = os.path.join(args.output_dir, "eval_images", f"step_{step}")
        os.makedirs(images_dir, exist_ok=True)

        for img_idx, (z_packed, caption) in enumerate(zip(all_z_packed, all_captions)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(z_packed, args.h, args.w)
                latents = (latents / 0.3611) + 0.1159
                image = vae.decode(latents, return_dict=False)[0]
                decoded = image_processor.postprocess(image)

            # Save locally
            img_path = os.path.join(images_dir, f"eval_{img_idx}.png")
            decoded[0].save(img_path)
            saved_paths.append(img_path)

            wandb_images.append(wandb.Image(decoded[0], caption=caption[:100]))

        wandb.log({"evaluation/samples": wandb_images}, step=step)
        logger.info(f"Eval images logged to wandb and saved to {images_dir}")

    transformer.train()
    dist.barrier()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Online DiffusionDPO for FLUX.1-dev with LoRA")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_dpo_flux")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--sampling_steps", type=int, default=8)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--beta_dpo", type=float, default=5000.0)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--hps_ckpt_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default="flux-dpo")
    parser.add_argument("--eval_every", type=int, default=0,
                        help="Eval and log images every N steps. 0 = only at checkpoints.")
    parser.add_argument("--max_eval_images", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Distributed setup ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
    )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Args: {vars(args)}")

    # --- Wandb ---
    if rank == 0:
        wandb.init(project=args.wandb_project, config=vars(args))

    # --- Seed ---
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    # --- Load models ---
    transformer, vae, scheduler = load_flux_models(args, device)

    if args.gradient_checkpointing:
        # Diffusers uses enable_gradient_checkpointing(), not gradient_checkpointing_enable().
        # Reach through PeftModel → LoraModel → FluxTransformer2DModel.
        transformer.base_model.model.enable_gradient_checkpointing()

    # --- FSDP ---
    transformer = setup_fsdp(transformer, device, args)

    # --- Optimizer ---
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.0)

    # --- Dataset ---
    dataset = LatentDataset(args.data_json_path)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # --- HPSv2 model ---
    hps_model, hps_preprocess, hps_tokenizer = load_hps_model(args.hps_ckpt_dir, device)

    # --- Compute tracker ---
    from utils.compute_tracker import DPOComputeTracker
    compute_tracker = DPOComputeTracker(num_gpus=world_size)
    if rank == 0:
        logger.info("Calibrating compute tracker...")
    calibrate_compute_tracker(
        compute_tracker, transformer, vae, hps_model, hps_tokenizer,
        device, args,
    )

    # --- Training ---
    global_step = 0
    optimizer.zero_grad()
    reward_log = []

    if rank == 0:
        logger.info("Starting training...")
        logger.info(f"  Num prompts = {len(dataset)}")
        logger.info(f"  Batch size per GPU = {args.train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {args.train_batch_size * world_size * args.gradient_accumulation_steps}")
        logger.info(f"  Num generations per prompt = {args.num_generations}")
        logger.info(f"  Max train steps = {args.max_train_steps}")

    # --- Eval baseline images before training (step 0) ---
    if rank == 0:
        logger.info("Generating baseline evaluation images (step 0)...")
    eval_and_log_images(args, transformer, vae, dataset, device,
                        step=0, eval_seed=args.seed, max_images=args.max_eval_images)

    step_times = []
    epoch = 0
    while global_step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            step_start = time.time()

            # === PHASE 1: Generate K images per prompt, score, select pair ===
            transformer.eval()
            with torch.no_grad():
                latents_all = generate_latents(transformer, scheduler, batch, args, device)
                # latents_all: (B*K, C, lh, lw)

                B = batch["prompt_embeds"].shape[0]
                K = args.num_generations

                # Decode to pixels for scoring
                pil_images = decode_latents(vae, latents_all)

                # Score with HPSv2
                prompts_repeated = []
                for p in batch["prompt"]:
                    prompts_repeated.extend([p] * K)
                scores = score_images(hps_model, hps_preprocess, hps_tokenizer,
                                      pil_images, prompts_repeated, device)

                # Select best/worst per prompt
                latents_w_list = []
                latents_l_list = []
                batch_rewards = []
                best_scores = []
                worst_scores = []
                for b_idx in range(B):
                    start = b_idx * K
                    end = start + K
                    prompt_scores = scores[start:end]
                    best_local = int(np.argmax(prompt_scores))
                    worst_local = int(np.argmin(prompt_scores))
                    best_idx = start + best_local
                    worst_idx = start + worst_local
                    latents_w_list.append(latents_all[best_idx])
                    latents_l_list.append(latents_all[worst_idx])
                    batch_rewards.append(float(np.mean(prompt_scores)))
                    best_scores.append(prompt_scores[best_local])
                    worst_scores.append(prompt_scores[worst_local])

                latents_w = torch.stack(latents_w_list)  # (B, C, lh, lw)
                latents_l = torch.stack(latents_l_list)  # (B, C, lh, lw)

            # === PHASE 2: DPO training step ===
            transformer.train()
            loss, metrics = dpo_training_step(
                transformer, vae, batch, latents_w, latents_l, args, device,
            )

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # Step optimizer
            grad_norm = 0.0
            if (global_step + 1) % args.gradient_accumulation_steps == 0 or \
               global_step == args.max_train_steps - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad()

            step_time = time.time() - step_start
            step_times.append(step_time)

            # === Record compute ===
            num_samples = B * K
            compute_tracker.record_sampling(num_samples, args.sampling_steps)
            compute_tracker.record_vae_decode(num_samples)
            compute_tracker.record_reward(num_samples)
            compute_tracker.record_dpo_training(B)

            # === Logging ===
            mean_reward = float(np.mean(batch_rewards))
            reward_log.append(mean_reward)

            if rank == 0 and global_step % args.log_every == 0:
                avg_step_time = sum(step_times) / len(step_times)
                logger.info(
                    f"Step {global_step}/{args.max_train_steps} | "
                    f"DPO Loss: {metrics['dpo_loss']:.4f} | "
                    f"Implicit Acc: {metrics['implicit_acc']:.4f} | "
                    f"Model MSE: {metrics['model_mse']:.6f} | "
                    f"Ref MSE: {metrics['ref_mse']:.6f} | "
                    f"Mean Reward: {mean_reward:.4f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"Time: {step_time:.1f}s"
                )

                # Wandb metrics
                log_dict = {
                    "train/dpo_loss": metrics["dpo_loss"],
                    "train/implicit_acc": metrics["implicit_acc"],
                    "train/model_mse": metrics["model_mse"],
                    "train/ref_mse": metrics["ref_mse"],
                    "train/mean_reward": mean_reward,
                    "train/best_reward": float(np.mean(best_scores)),
                    "train/worst_reward": float(np.mean(worst_scores)),
                    "train/reward_gap": float(np.mean(best_scores)) - float(np.mean(worst_scores)),
                    "train/grad_norm": grad_norm,
                    "train/step_time": step_time,
                    "train/avg_step_time": avg_step_time,
                    "train/learning_rate": args.learning_rate,
                }
                log_dict.update(compute_tracker.get_metrics())
                wandb.log(log_dict, step=global_step)

            # === Save rewards ===
            if rank == 0 and global_step % args.log_every == 0:
                reward_path = os.path.join(args.output_dir, "reward.txt")
                with open(reward_path, "a") as f:
                    f.write(f"{global_step}\t{mean_reward:.6f}\t{metrics['dpo_loss']:.6f}\t"
                            f"{metrics['implicit_acc']:.4f}\n")

            # === Checkpointing + eval images ===
            if (global_step + 1) % args.checkpointing_steps == 0:
                # All ranks must participate in FSDP state dict gathering
                save_lora_checkpoint(transformer, optimizer, global_step + 1, args.output_dir)
                # Log eval images at each checkpoint (cfgrl-expo style)
                eval_and_log_images(args, transformer, vae, dataset, device,
                                    step=global_step + 1, eval_seed=args.seed,
                                    max_images=args.max_eval_images)
                dist.barrier()

            # === Optional periodic eval ===
            elif args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
                eval_and_log_images(args, transformer, vae, dataset, device,
                                    step=global_step + 1, eval_seed=args.seed,
                                    max_images=args.max_eval_images)
                dist.barrier()

            global_step += 1

        epoch += 1

    # Final save + eval (all ranks participate in FSDP state dict gathering)
    save_lora_checkpoint(transformer, optimizer, global_step, args.output_dir)

    eval_and_log_images(args, transformer, vae, dataset, device,
                        step=global_step, eval_seed=args.seed,
                        max_images=args.max_eval_images)

    # Print and save compute summary
    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("COMPUTE SUMMARY")
        logger.info("=" * 60)
        logger.info(compute_tracker.summary())

        summary_path = os.path.join(args.output_dir, "compute_summary.txt")
        with open(summary_path, "w") as f:
            f.write(compute_tracker.summary() + "\n")
            f.write("\nCalibrated per-operation FLOPs:\n")
            f.write(f"  sampling_step_flops: {compute_tracker.sampling_step_flops:,}\n")
            f.write(f"  vae_decode_flops: {compute_tracker.vae_decode_flops:,}\n")
            f.write(f"  reward_forward_flops: {compute_tracker.reward_forward_flops:,}\n")
            f.write(f"  training_step_flops: {compute_tracker.training_step_flops:,}\n")
        logger.info(f"Compute summary saved to {summary_path}")
        logger.info("Training complete!")

        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
