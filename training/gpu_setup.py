"""GPU auto-detection, config overrides, and mixed-precision utilities for Colab training."""

from __future__ import annotations

import gc
import os
from typing import Optional

import torch

from configs import STAGE1_DEFAULT_CONFIG, STAGE2_DEFAULT_CONFIG


def detect_gpu_setup(force_batch_size: Optional[int] = None) -> None:
    """Auto-detect GPU specs and override configs for 4-hour Colab Pro training.

    Overrides ``STAGE1_DEFAULT_CONFIG`` and ``STAGE2_DEFAULT_CONFIG`` in-place
    (via ``object.__setattr__`` on the frozen dataclass) based on available GPU
    VRAM.  On A100 80GB+ systems the schedule is compressed to 30+20 epochs;
    on smaller GPUs the default 50+30 schedule is retained.

    Args:
        force_batch_size: If provided, use this batch size regardless of VRAM.
    """
    if not torch.cuda.is_available():
        print("No GPU detected. Using CPU defaults.")
        return

    num_gpus = torch.cuda.device_count()
    vram_gb: list[float] = []
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        vram_gb.append(vram)
        print(f"  GPU {i}: {name} | {vram:.1f} GB VRAM")

    total_vram = sum(vram_gb)
    peak_vram = max(vram_gb)
    print(f"\nDetected: {num_gpus} GPU(s), total VRAM = {total_vram:.1f} GB")

    if force_batch_size is not None:
        bs = force_batch_size
    elif peak_vram >= 75:
        bs = 64
    elif peak_vram >= 40:
        bs = 32
    elif peak_vram >= 16:
        bs = 16
    else:
        bs = 8

    if num_gpus > 1:
        bs = bs * num_gpus

    cpu_count = os.cpu_count() or 4
    n_workers = min(8, cpu_count)

    print(f"  -> Stage 1 batch_size = {bs}")
    print(f"  -> Stage 2 batch_size = {min(bs * 2, 128)}")
    print(f"  -> num_workers = {n_workers}")

    object.__setattr__(STAGE1_DEFAULT_CONFIG, "batch_size", bs)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "num_workers", n_workers)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "multi_gpu", num_gpus > 1)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "samples_per_epoch_multiplier", 3.0)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "use_fp16", True)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "resume", None)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "epochs", 30)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "focal_start_epoch", 6)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "focal_full_epoch", 10)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "sc_dfa_start_epoch", 9)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "sc_dfa_full_epoch", 13)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "prior_start_epoch", 12)
    object.__setattr__(STAGE1_DEFAULT_CONFIG, "prior_full_epoch", 17)
    print(
        f"  -> Stage 1: epochs={STAGE1_DEFAULT_CONFIG.epochs}"
        f", focal={STAGE1_DEFAULT_CONFIG.focal_start_epoch}->{STAGE1_DEFAULT_CONFIG.focal_full_epoch}"
        f", sc_dfa={STAGE1_DEFAULT_CONFIG.sc_dfa_start_epoch}->{STAGE1_DEFAULT_CONFIG.sc_dfa_full_epoch}"
        f", prior={STAGE1_DEFAULT_CONFIG.prior_start_epoch}->{STAGE1_DEFAULT_CONFIG.prior_full_epoch}"
    )

    object.__setattr__(STAGE2_DEFAULT_CONFIG, "batch_size", min(bs * 2, 128))
    object.__setattr__(STAGE2_DEFAULT_CONFIG, "num_workers", n_workers)
    object.__setattr__(STAGE2_DEFAULT_CONFIG, "epochs", 20)
    object.__setattr__(STAGE2_DEFAULT_CONFIG, "keep_lambda_decay_epochs", 20)
    object.__setattr__(STAGE2_DEFAULT_CONFIG, "alpha_warmup_epochs", 20)
    object.__setattr__(STAGE2_DEFAULT_CONFIG, "samples_per_epoch_multiplier", 4.0)
    object.__setattr__(STAGE2_DEFAULT_CONFIG, "use_fp16", True)
    print(
        f"  -> Stage 2: epochs={STAGE2_DEFAULT_CONFIG.epochs}"
        f", keep_lambda_decay={STAGE2_DEFAULT_CONFIG.keep_lambda_decay_epochs}"
        f", alpha_warmup={STAGE2_DEFAULT_CONFIG.alpha_warmup_epochs}"
    )

    if torch.cuda.is_bf16_supported():
        print("  -> bfloat16 supported: mixed precision will use bf16")
    else:
        print("  -> Using float16 mixed precision")

    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print("  -> TF32 Tensor Cores enabled, matmul precision=high")

    cleanup_gpu_cache()
    print(f"\nGPU cache cleared. Available VRAM: {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")


def patch_autocast_for_bf16() -> bool:
    """Patch ``training.train_loop._autocast_context`` to use bf16 on Ampere+.

    On CUDA devices that support bfloat16 (Ampere/Hopper), this replaces the
    default FP16 autocast context with bf16 for more stable mixed-precision
    training.  On CPU or pre-Ampere GPUs this is a no-op.

    Returns:
        ``True`` if bf16 was enabled, ``False`` otherwise.
    """
    import training.train_loop as tl

    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        print("Using bfloat16: False")
        return False

    tl._autocast_context = lambda device: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    print("Using bfloat16: True")
    return True


def cleanup_gpu_cache() -> None:
    """Run garbage collection and clear PyTorch CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


__all__ = [
    "cleanup_gpu_cache",
    "detect_gpu_setup",
    "patch_autocast_for_bf16",
]
