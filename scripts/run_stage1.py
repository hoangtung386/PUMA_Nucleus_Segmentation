#!/usr/bin/env python3
"""Entry point for Stage 1 training."""

from training.gpu_setup import detect_gpu_setup
from training.stage1_trainer import main

if __name__ == "__main__":
    cfg = detect_gpu_setup()
    main(cfg)
