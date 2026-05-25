"""GPU auto-detection, config overrides, and mixed-precision utilities."""

from __future__ import annotations

import gc
import os
from dataclasses import replace
from typing import Optional

import torch

from configs import STAGE1_DEFAULT_CONFIG
from configs.defaults import Stage1Config


def detect_gpu_setup(force_batch_size: Optional[int] = None) -> Stage1Config:
    if not torch.cuda.is_available():
        print("No GPU detected. Using CPU defaults.")
        return STAGE1_DEFAULT_CONFIG

    num_gpus = torch.cuda.device_count()
    vram_gb: list[float] = []
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        vram_gb.append(vram)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | {vram:.1f} GB VRAM")

    peak_vram = max(vram_gb)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if force_batch_size is not None:
        bs = force_batch_size
    elif peak_vram >= 75:
        bs = 12
    elif peak_vram >= 40:
        bs = 8
    elif peak_vram >= 16:
        bs = 4
    else:
        bs = 2

    if num_gpus > 1:
        bs = bs * num_gpus

    cpu_count = os.cpu_count() or 4
    n_workers = min(4, cpu_count)

    print(f"  -> batch_size = {bs}, num_workers = {n_workers}")

    cfg = replace(
        STAGE1_DEFAULT_CONFIG,
        batch_size=bs,
        num_workers=n_workers,
        multi_gpu=num_gpus > 1,
        use_fp16=True,
        resume=None,
        epochs=30,
        focal_start_epoch=6,
        focal_full_epoch=10,
        sc_dfa_start_epoch=9,
        sc_dfa_full_epoch=13,
    )

    if torch.cuda.is_bf16_supported():
        print("  -> bfloat16 supported")
    else:
        print("  -> Using float16 mixed precision")

    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print("  -> TF32 Tensor Cores enabled")

    cleanup_gpu_cache()
    return cfg


def patch_autocast_for_bf16() -> bool:
    import training.train_loop as tl

    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        print("Using bfloat16: False")
        return False

    tl._autocast_context = lambda device: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    print("Using bfloat16: True")
    return True


def cleanup_gpu_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


__all__ = ["cleanup_gpu_cache", "detect_gpu_setup", "patch_autocast_for_bf16"]
