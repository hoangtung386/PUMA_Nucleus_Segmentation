"""CLI argument parsing for Stage 1 training — v8."""

import argparse
from dataclasses import replace

from configs.defaults import Stage1Config


def parse_stage1_args() -> Stage1Config:
    parser = argparse.ArgumentParser(description="Stage 1 Training (v8 CellPath)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use-context-encoder", action="store_true", default=None)
    parser.add_argument("--use-stain-aug", action="store_true", default=None)
    args = parser.parse_args()

    config = Stage1Config()
    overrides: dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.val_ratio is not None:
        overrides["val_ratio"] = args.val_ratio
    if args.resume is not None:
        overrides["resume"] = args.resume
    if args.use_context_encoder is not None:
        overrides["use_context_encoder"] = args.use_context_encoder
    if args.use_stain_aug is not None:
        overrides["use_stain_aug"] = args.use_stain_aug
    return replace(config, **overrides) if overrides else config
