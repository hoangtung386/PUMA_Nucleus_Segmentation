"""GPU auto-detection, config overrides, and mixed-precision utilities."""

from __future__ import annotations

import gc
import os
from dataclasses import replace

import torch

from configs import STAGE1_DEFAULT_CONFIG
from configs.defaults import Stage1Config
from symbiopan.common.logging import get_logger

logger = get_logger(__name__)


def detect_gpu_setup(force_batch_size: int | None = None) -> Stage1Config:
    if not torch.cuda.is_available():
        logger.info("No GPU detected. Using CPU defaults.")
        return STAGE1_DEFAULT_CONFIG

    num_gpus = torch.cuda.device_count()
    vram_gb: list[float] = []
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        vram_gb.append(vram)
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} | {vram:.1f} GB VRAM")

    peak_vram = max(vram_gb)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if force_batch_size is not None:
        bs = force_batch_size
    elif peak_vram >= 75:
        bs = 6
    elif peak_vram >= 40:
        bs = 4
    elif peak_vram >= 16:
        bs = 2
    else:
        bs = 1

    if num_gpus > 1:
        bs = bs * num_gpus

    # Effective batch ≈ 12–16; compensate smaller bs with grad_accum
    accum = max(2, 12 // bs)

    cpu_count = os.cpu_count() or 4
    n_workers = min(8, cpu_count)

    logger.info(f"  -> batch_size = {bs}, grad_accum_steps = {accum}, num_workers = {n_workers}")

    cfg = replace(
        STAGE1_DEFAULT_CONFIG,
        batch_size=bs,
        grad_accum_steps=accum,
        num_workers=n_workers,
        multi_gpu=num_gpus > 1,
        use_fp16=True,
        compile_model=torch.cuda.get_device_capability() >= (7, 0),
        resume=None,
        epochs=30,
        focal_start_epoch=6,
        focal_full_epoch=10,
        sc_dfa_start_epoch=9,
        sc_dfa_full_epoch=13,
    )

    if torch.cuda.is_bf16_supported():
        logger.info("  -> bfloat16 supported")
    else:
        logger.info("  -> Using float16 mixed precision")

    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    logger.info("  -> TF32 Tensor Cores enabled")

    cleanup_gpu_cache()
    return cfg


def cleanup_gpu_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


__all__ = ["cleanup_gpu_cache", "detect_gpu_setup"]
