"""Training compute (FLOP) tracker for DiffusionDPO experiments.

Uses PyTorch's built-in FlopCounterMode to measure actual FLOPs for each
phase of the online DPO training loop, then accumulates totals over the run.

On the first training step, each phase (sampling, VAE decode, reward,
DPO training) is profiled with FlopCounterMode to obtain per-operation
FLOP counts. Subsequent steps multiply by operation counts to avoid
profiling overhead.

Metric names align with cfgrl-expo / DanceGRPO's ComputeTracker for
direct comparison.

DPO training step structure (per prompt):
  - Sampling:  K images * sampling_steps forward passes (no grad)
  - VAE:       K images decoded to pixels
  - Reward:    K images scored with HPSv2
  - DPO train: 2 fwd w/ grad (policy_w, policy_l)
             + 2 fwd no grad (ref_w, ref_l)
             + 1 backward pass

Usage:
    tracker = DPOComputeTracker(num_gpus=4)

    # First step: calibrate by wrapping each phase
    with tracker.calibrate_sampling():
        transformer(...)  # single forward pass
    with tracker.calibrate_vae_decode():
        vae.decode(...)
    with tracker.calibrate_reward():
        reward_model(...)
    with tracker.calibrate_training():
        # fwd + bwd for one sample
        pred = transformer(...)
        loss = pred.sum()
        loss.backward()

    # All steps: record operation counts
    tracker.record_sampling(num_samples, sampling_steps)
    tracker.record_vae_decode(num_samples)
    tracker.record_reward(num_samples)
    tracker.record_dpo_training(batch_size)

    # Log
    metrics = tracker.get_metrics()
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.utils.flop_counter import FlopCounterMode


@dataclass
class DPOComputeTracker:
    """Tracks cumulative training FLOPs using measured per-op costs.

    All FLOP values are raw FLOPs (1 FMA = 2 FLOPs), matching PyTorch's
    FlopCounterMode convention.
    """

    num_gpus: int = 1

    # Per-operation FLOP costs (set during calibration).
    # These are per-GPU, single-invocation costs.
    sampling_step_flops: int = 0       # One FLUX forward pass (one denoising step)
    vae_decode_flops: int = 0          # One VAE decode call
    reward_forward_flops: int = 0      # One reward model forward pass
    training_step_flops: int = 0       # One transformer fwd+bwd (one sample)

    # Calibration status.
    _calibrated: dict = field(default_factory=dict, repr=False)

    # Cumulative counters (per-GPU FLOPs).
    total_sampling_flops: int = field(default=0, repr=False)
    total_vae_flops: int = field(default=0, repr=False)
    total_reward_flops: int = field(default=0, repr=False)
    total_training_flops: int = field(default=0, repr=False)

    _start_time: float = field(default_factory=time.time, repr=False)

    # -- Calibration context managers ------------------------------------------

    @contextmanager
    def calibrate_sampling(self):
        """Wrap a single FLUX forward pass (one denoising step) to measure FLOPs."""
        counter = FlopCounterMode(display=False)
        with counter:
            yield
        self.sampling_step_flops = counter.get_total_flops()
        self._calibrated["sampling"] = True
        print(f"[ComputeTracker] Calibrated sampling: {self.sampling_step_flops:,} FLOPs/step")

    @contextmanager
    def calibrate_vae_decode(self):
        """Wrap a single VAE decode call to measure FLOPs."""
        counter = FlopCounterMode(display=False)
        with counter:
            yield
        self.vae_decode_flops = counter.get_total_flops()
        self._calibrated["vae_decode"] = True
        print(f"[ComputeTracker] Calibrated VAE decode: {self.vae_decode_flops:,} FLOPs/call")

    @contextmanager
    def calibrate_reward(self):
        """Wrap a single reward model forward pass to measure FLOPs."""
        counter = FlopCounterMode(display=False)
        with counter:
            yield
        self.reward_forward_flops = counter.get_total_flops()
        self._calibrated["reward"] = True
        print(f"[ComputeTracker] Calibrated reward: {self.reward_forward_flops:,} FLOPs/call")

    @contextmanager
    def calibrate_training(self):
        """Wrap a single transformer fwd+bwd pass to measure FLOPs."""
        counter = FlopCounterMode(display=False)
        with counter:
            yield
        self.training_step_flops = counter.get_total_flops()
        self._calibrated["training"] = True
        print(f"[ComputeTracker] Calibrated training: {self.training_step_flops:,} FLOPs/step")

    @property
    def is_calibrated(self) -> bool:
        return all(k in self._calibrated for k in
                   ["sampling", "vae_decode", "reward", "training"])

    def print_calibration_summary(self):
        """Print a summary of calibrated FLOP costs."""
        components = ["sampling", "vae_decode", "reward", "training"]
        n_cal = sum(1 for c in components if c in self._calibrated)
        print(f"[ComputeTracker] Calibrated {n_cal}/{len(components)} components:")
        flop_attrs = {
            "sampling": "sampling_step_flops",
            "vae_decode": "vae_decode_flops",
            "reward": "reward_forward_flops",
            "training": "training_step_flops",
        }
        for c in components:
            if c in self._calibrated:
                val = getattr(self, flop_attrs[c])
                print(f"  {c}: {val:,} FLOPs ({val / 1e12:.4f} TFLOP)")
            else:
                print(f"  {c}: NOT CALIBRATED")

    # -- Recording methods (call every step) -----------------------------------

    def record_sampling(self, num_samples: int, sampling_steps: int):
        """Record the sampling/rollout phase.

        Each sample requires sampling_steps forward passes through FLUX.
        """
        self.total_sampling_flops += num_samples * sampling_steps * self.sampling_step_flops

    def record_vae_decode(self, num_samples: int):
        """Record VAE decoding of generated latents to images."""
        self.total_vae_flops += num_samples * self.vae_decode_flops

    def record_reward(self, num_samples: int):
        """Record reward model inference."""
        self.total_reward_flops += num_samples * self.reward_forward_flops

    def record_dpo_training(self, batch_size: int):
        """Record the DPO gradient update phase.

        Per batch, DPO does:
          - 2 forward passes with grad (policy on preferred + unpreferred)
          - 2 forward passes no grad (reference on preferred + unpreferred)
          - 1 backward pass
        Approximation: 4 fwd + 1 bwd ≈ 4 fwd + 2 fwd = 6 fwd-equivalent passes
        But calibrate_training measures one fwd+bwd, so we scale by:
          - 2 fwd_no_grad ≈ 2 fwd ≈ 1 fwd+bwd (since bwd ≈ 2x fwd)
          - 2 fwd_with_grad + bwd ≈ 1 fwd+bwd (already measured)
        Total ≈ 2 × training_step_flops per batch item
        """
        self.total_training_flops += batch_size * 2 * self.training_step_flops

    # -- Derived metrics -------------------------------------------------------

    @property
    def total_inference_flops(self) -> int:
        return self.total_sampling_flops + self.total_vae_flops + self.total_reward_flops

    @property
    def total_flops_per_gpu(self) -> int:
        return self.total_inference_flops + self.total_training_flops

    @property
    def total_flops(self) -> int:
        return self.total_flops_per_gpu * self.num_gpus

    @property
    def wall_clock_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def gpu_hours(self) -> float:
        return self.wall_clock_seconds * self.num_gpus / 3600.0

    def get_metrics(self) -> dict:
        """Return a dict of compute metrics for wandb logging."""
        total = self.total_flops
        return {
            "compute/total_flops": total,
            "compute/total_tflops": total / 1e12,
            "compute/sampling_flops": self.total_sampling_flops * self.num_gpus,
            "compute/sampling_tflops": self.total_sampling_flops * self.num_gpus / 1e12,
            "compute/training_flops": self.total_training_flops * self.num_gpus,
            "compute/training_tflops": self.total_training_flops * self.num_gpus / 1e12,
            "compute/vae_flops": self.total_vae_flops * self.num_gpus,
            "compute/reward_flops": self.total_reward_flops * self.num_gpus,
            "compute/inference_flops": self.total_inference_flops * self.num_gpus,
            "compute/inference_tflops": self.total_inference_flops * self.num_gpus / 1e12,
            "compute/wall_clock_seconds": self.wall_clock_seconds,
            "compute/gpu_hours": self.gpu_hours,
        }

    def summary(self) -> str:
        """Human-readable compute summary."""
        total = self.total_flops
        inf = self.total_inference_flops * self.num_gpus
        trn = self.total_training_flops * self.num_gpus
        lines = [
            f"Total: {total / 1e12:.2f} TFLOP ({self.num_gpus} GPUs)",
            f"  Sampling:  {self.total_sampling_flops * self.num_gpus / 1e12:.2f} TFLOP",
            f"  VAE:       {self.total_vae_flops * self.num_gpus / 1e12:.4f} TFLOP",
            f"  Reward:    {self.total_reward_flops * self.num_gpus / 1e12:.4f} TFLOP",
            f"  Inference: {inf / 1e12:.2f} TFLOP ({100 * inf / max(total, 1):.1f}%)",
            f"  Training:  {trn / 1e12:.2f} TFLOP ({100 * trn / max(total, 1):.1f}%)",
            f"  Wall:      {self.wall_clock_seconds:.0f}s ({self.gpu_hours:.2f} GPU-hours)",
        ]
        if self._calibrated:
            for name in self._calibrated:
                lines.append(f"  {name}: measured")
        return "\n".join(lines)
