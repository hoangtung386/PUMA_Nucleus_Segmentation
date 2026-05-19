"""CLI argument parsing for training stages."""

import argparse
from dataclasses import replace

from configs.defaults import Stage1Config, Stage2Config


def parse_stage1_args() -> Stage1Config:
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = Stage1Config()
    overrides = {}
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
    return replace(config, **overrides) if overrides else config


def parse_stage2_args() -> Stage2Config:
    parser = argparse.ArgumentParser(description="Stage 2 Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    args = parser.parse_args()

    config = Stage2Config()
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.val_ratio is not None:
        overrides["val_ratio"] = args.val_ratio
    return replace(config, **overrides) if overrides else config
