"""
Early-stopping and model-checkpoint callbacks for the training loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─── Early stopping ───────────────────────────────────────────────────────────


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:   Number of epochs with no improvement before stopping.
        min_delta:  Minimum change to qualify as an improvement.
        mode:       ``"min"`` (loss) or ``"max"`` (accuracy / F1).
        verbose:    Log a message each time the counter advances.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter: int = 0
        self.best_value: Optional[float] = None
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Call after each epoch.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: no improvement for %d/%d epoch(s). Best=%.4f",
                    self.counter,
                    self.patience,
                    self.best_value,
                )
            if self.counter >= self.patience:
                logger.info("EarlyStopping: triggered. Training stopped.")
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        self.counter = 0
        self.best_value = None
        self.should_stop = False


# ─── Model checkpoint ─────────────────────────────────────────────────────────


class ModelCheckpoint:
    """Save the model whenever a monitored metric improves.

    Args:
        save_dir:    Directory to store checkpoints.
        filename:    Checkpoint filename stem (``".pth"`` appended automatically).
        monitor:     Metric name to watch (for logging only).
        mode:        ``"min"`` or ``"max"``.
        save_best_only: If ``True``, always overwrite the previous best.
                        If ``False``, save a new file each epoch.
        verbose:     Log a message on each save.
    """

    def __init__(
        self,
        save_dir: str | Path,
        filename: str = "best_model",
        monitor: str = "val_f1",
        mode: str = "max",
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_value: Optional[float] = None
        self.best_path: Optional[Path] = None

    def step(
        self,
        model: nn.Module,
        value: float,
        epoch: int,
        extra: Optional[dict] = None,
    ) -> bool:
        """Save the model if the monitored metric improved.

        Args:
            model:  The PyTorch model to save.
            value:  Current metric value.
            epoch:  Current epoch index.
            extra:  Extra dict to store alongside the state_dict
                    (e.g. optimizer state, config).

        Returns:
            ``True`` if the checkpoint was saved.
        """
        improved = self.best_value is None or (
            value < self.best_value if self.mode == "min" else value > self.best_value
        )

        if not improved:
            return False

        self.best_value = value

        if self.save_best_only:
            path = self.save_dir / f"{self.filename}.pth"
        else:
            path = self.save_dir / f"{self.filename}_epoch{epoch:04d}.pth"

        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            self.monitor: value,
        }
        if extra:
            payload.update(extra)

        torch.save(payload, str(path))
        self.best_path = path

        if self.verbose:
            logger.info(
                "ModelCheckpoint: %s improved to %.4f → saved to %s",
                self.monitor,
                value,
                path,
            )
        return True

    def restore_best(self, model: nn.Module) -> None:
        """Load the best checkpoint into ``model``."""
        if self.best_path is None or not self.best_path.exists():
            logger.warning("ModelCheckpoint: no checkpoint to restore.")
            return
        ckpt = torch.load(str(self.best_path), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Restored best checkpoint from %s", self.best_path)
