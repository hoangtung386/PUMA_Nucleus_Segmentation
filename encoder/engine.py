from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ENCODER_RUNS, ENCODER_RUN_ALIASES, OUTPUT_DIR, TrainConfig, cfg_for_fold
from .data import PumaPatchDataset, make_split, seed_everything
from .losses import PumaMultiTaskLoss
from .metrics import PumaMetrics
from .models import PumaEncoderProbe
from .splits import generate_multilabel_folds


def _initialize_lazy_modules(
    model: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
) -> None:
    """
    Initialize LazyConv2d/LazyLinear-style layers before parameter counting
    and optimizer construction.

    PumaEncoderProbe uses lazy projection layers because the feature-channel
    count can differ across UNIv2, Virchow2, ConvNeXt, and fusion experiments.

    Lazy parameters have no shape until the first forward pass, so calling
    p.numel() before this dummy forward causes:

        ValueError: Attempted to use an uninitialized parameter
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(
            1,
            3,
            cfg.image_size,
            cfg.image_size,
            device=device,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            _ = model(dummy)

    model.train(was_training)


def _count_initialized_params(params) -> int:
    """
    Count parameters safely.

    Uninitialized lazy parameters are skipped. After _initialize_lazy_modules(),
    normally everything should already be initialized, but this function keeps
    the code safe.
    """
    total = 0

    for p in params:
        if isinstance(p, UninitializedParameter):
            continue

        total += p.numel()

    return total


def _move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()

    totals: Dict[str, float] = {}
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch["image"])
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())

        steps += 1

    return {
        k: v / max(1, steps)
        for k, v in totals.items()
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    totals: Dict[str, float] = {}
    metrics = PumaMetrics()
    steps = 0

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch["image"])
            loss_dict = criterion(outputs, batch)

        metrics.update(outputs, batch)

        for k, v in loss_dict.items():
            totals[f"val_{k}"] = totals.get(f"val_{k}", 0.0) + float(v.detach().cpu())

        steps += 1

    out = {
        k: v / max(1, steps)
        for k, v in totals.items()
    }

    out.update(metrics.compute())

    return out


def run_training(cfg: TrainConfig) -> Path:
    generate_multilabel_folds(cfg, force=False)
    seed_everything(cfg.seed + cfg.fold_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.amp and device.type == "cuda")

    split = make_split(cfg)

    train_ds = PumaPatchDataset(split.train, cfg, train=True)
    val_ds = PumaPatchDataset(split.val, cfg, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    out_dir = OUTPUT_DIR / cfg.experiment_name / f"fold_{cfg.fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    split.train.to_csv(out_dir / "train_split.csv", index=False)
    split.val.to_csv(out_dir / "val_split.csv", index=False)

    model = PumaEncoderProbe(cfg).to(device)
    criterion = PumaMultiTaskLoss().to(device)

    # Important fix:
    # The probe uses lazy projection layers. They must be initialized before
    # parameter counting and before optimizer construction.
    _initialize_lazy_modules(
        model=model,
        cfg=cfg,
        device=device,
        use_amp=use_amp,
    )

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found. Projection/head layers should remain trainable."
        )

    trainable_param_count = _count_initialized_params(trainable_params)
    total_param_count = _count_initialized_params(model.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -math.inf
    bad_epochs = 0
    history = []

    print(f"Running {cfg.experiment_name} fold {cfg.fold_id}/{cfg.n_folds - 1}")
    print(f"Train images={len(split.train)} | Val images={len(split.val)} | output={out_dir}")
    print(f"Epochs={cfg.epochs} | Batch size={cfg.batch_size}")
    print(f"Trainable params={trainable_param_count:,} / Total params={total_param_count:,}")

    for epoch in range(1, cfg.epochs + 1):
        train_log = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        val_log = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        row = {
            "epoch": epoch,
            "fold": cfg.fold_id,
            **train_log,
            **val_log,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            out_dir / "metrics.csv",
            index=False,
        )

        print(json.dumps(row, indent=2))

        # Official checkpoint selection rule.
        # Do not use old selection_score here.
        score = float(val_log.get("official_selection_score", 0.0))

        if score > best_score:
            best_score = score
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_score": best_score,
                    "best_metric": "official_selection_score",
                    "metrics": val_log,
                },
                out_dir / "best_model.pth",
            )

            print(
                f"Saved new best checkpoint at epoch {epoch} "
                f"with official_selection_score={best_score:.6f}"
            )

        else:
            bad_epochs += 1

            if bad_epochs >= cfg.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best official_selection_score={best_score:.6f}"
                )
                break

    return out_dir


def run_by_name(
    name: str,
    fold: int | None = None,
    all_folds: bool = False,
) -> list[Path]:
    canonical = ENCODER_RUN_ALIASES.get(name, name)

    if canonical not in ENCODER_RUNS:
        raise KeyError(
            f"Unknown run {name}. Available: {list(ENCODER_RUNS)}"
        )

    base_cfg = ENCODER_RUNS[canonical]

    generate_multilabel_folds(base_cfg, force=False)

    if all_folds or fold is None:
        fold_ids = list(range(base_cfg.n_folds))
    else:
        fold_ids = [fold]

    outputs = []

    for fold_id in fold_ids:
        outputs.append(
            run_training(
                cfg_for_fold(base_cfg, fold_id)
            )
        )

    return outputs