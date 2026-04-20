"""
Training loop implementations for the PUMA pipeline.

Two trainers are provided:

SegmentationTrainer
    Delegates to Cellpose's ``train_seg`` API. Manages data loading,
    epoch loop, TensorBoard logging, and checkpoint saving.

ClassificationTrainer
    Standard PyTorch training loop for the ``NucleusClassifier``.
    Supports 2-phase training (frozen backbone → gradual unfreeze) and
    integrates with the ``EarlyStopping`` / ``ModelCheckpoint`` callbacks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from puma_seg.models.cellpose_wrapper import CellposeSegmentor
from puma_seg.models.losses import ClassificationLoss
from puma_seg.models.nucleus_classifier import NucleusClassifier
from puma_seg.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


# ─── Segmentation trainer ──────────────────────────────────────────────────────


class SegmentationTrainer:
    """Fine-tune Cellpose segmentation using the official ``train_seg`` API.

    Args:
        segmentor:       A ``CellposeSegmentor`` instance (wraps CellposeModel).
        save_dir:        Directory to store fine-tuned checkpoints.
        log_dir:         TensorBoard log directory.
        model_name:      Filename stem for the saved checkpoint.
    """

    def __init__(
        self,
        segmentor: CellposeSegmentor,
        save_dir: str | Path = "./models",
        log_dir: str | Path = "./runs/segmentation",
        model_name: str = "puma_cellpose",
    ) -> None:
        self.segmentor = segmentor
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def train(
        self,
        train_images: List,
        train_labels: List,
        val_images: Optional[List] = None,
        val_labels: Optional[List] = None,
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 8,
        channels: Optional[List[int]] = None,
    ) -> Dict:
        """Run segmentation fine-tuning and log losses to TensorBoard.

        Args:
            train_images: List of np.ndarray (H, W, 3) uint8.
            train_labels: List of np.ndarray (H, W) int32 instance masks.
            val_images:   Optional validation images.
            val_labels:   Optional validation labels.
            learning_rate: Fine-tune LR (keep ≤ 1e-5).
            weight_decay:  AdamW weight decay.
            n_epochs:      Training epochs.
            batch_size:    Batch size.
            channels:      Cellpose channel config (default [0, 0]).

        Returns:
            dict with ``model_path``, ``train_losses``, ``test_losses``.
        """
        logger.info(
            "SegmentationTrainer: starting %d epochs | lr=%.1e | %d train / %d val images",
            n_epochs,
            learning_rate,
            len(train_images),
            len(val_images) if val_images else 0,
        )

        model_path, train_losses, test_losses = self.segmentor.fine_tune(
            train_images=train_images,
            train_labels=train_labels,
            test_images=val_images,
            test_labels=val_labels,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_name=self.model_name,
            save_path=self.save_dir,
            channels=channels,
        )

        # Log to TensorBoard
        for epoch, (t_loss, v_loss) in enumerate(
            zip(train_losses, test_losses or [None] * len(train_losses))
        ):
            self.writer.add_scalar("seg/train_loss", t_loss, epoch)
            if v_loss is not None:
                self.writer.add_scalar("seg/val_loss", v_loss, epoch)

        self.writer.close()
        return {
            "model_path": model_path,
            "train_losses": train_losses,
            "test_losses": test_losses,
        }


# ─── Classification trainer ────────────────────────────────────────────────────


class ClassificationTrainer:
    """Two-phase training loop for ``NucleusClassifier``.

    Phase 1 — frozen backbone:
        Only the classification head is trained.  Fast convergence.
    Phase 2 — gradual unfreeze:
        Unfreeze the last ``unfreeze_groups`` ResNet layer groups with a
        much smaller learning rate for fine-grained adaptation.

    Args:
        model:          ``NucleusClassifier`` instance.
        device:         ``"cuda"`` or ``"cpu"``.
        save_dir:       Directory to store checkpoints.
        log_dir:        TensorBoard log directory.
        track:          Challenge track (1 or 2) — used to select class count.
    """

    def __init__(
        self,
        model: NucleusClassifier,
        device: str = "cuda",
        save_dir: str | Path = "./models",
        log_dir: str | Path = "./runs/classifier",
        track: int = 1,
    ) -> None:
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_dir = Path(save_dir)
        self.track = track
        self.writer = SummaryWriter(log_dir=str(log_dir))

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        *,
        # Phase 1
        phase1_epochs: int = 20,
        phase1_lr: float = 1e-3,
        # Phase 2
        phase2_epochs: int = 80,
        phase2_lr_head: float = 1e-4,
        phase2_lr_backbone: float = 1e-6,
        unfreeze_groups: int = 2,
        # Common
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        patience: int = 15,
        use_amp: bool = True,
    ) -> Dict:
        """Run the full 2-phase training.

        Returns:
            dict with ``best_val_f1`` and ``checkpoint_path``.
        """
        loss_fn = ClassificationLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        checkpoint = ModelCheckpoint(
            save_dir=self.save_dir,
            filename="best_classifier",
            monitor="val_f1",
            mode="max",
        )
        early_stop = EarlyStopping(patience=patience, mode="max")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and self.device.type == "cuda")

        global_epoch = 0

        # ── Phase 1: frozen backbone ─────────────────────────────────────────
        logger.info("=== Phase 1: frozen backbone — %d epochs ===", phase1_epochs)
        self.model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=phase1_lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase1_epochs, eta_min=phase1_lr * 0.01
        )
        global_epoch = self._run_epochs(
            phase1_epochs,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            checkpoint,
            early_stop,
            scaler,
            global_epoch,
            phase="phase1",
            use_amp=use_amp,
        )

        # ── Phase 2: gradual unfreeze ────────────────────────────────────────
        logger.info(
            "=== Phase 2: unfreezing %d backbone group(s) — %d epochs ===",
            unfreeze_groups,
            phase2_epochs,
        )
        self.model.unfreeze_backbone(layer_groups=unfreeze_groups)
        early_stop.reset()

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.backbone.parameters(),
                    "lr": phase2_lr_backbone,
                },
                {
                    "params": self.model.head.parameters(),
                    "lr": phase2_lr_head,
                },
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase2_epochs, eta_min=phase2_lr_head * 0.01
        )
        self._run_epochs(
            phase2_epochs,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_fn,
            checkpoint,
            early_stop,
            scaler,
            global_epoch,
            phase="phase2",
            use_amp=use_amp,
        )

        self.writer.close()
        return {
            "best_val_f1": checkpoint.best_value,
            "checkpoint_path": checkpoint.best_path,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_epochs(
        self,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        checkpoint: ModelCheckpoint,
        early_stop: EarlyStopping,
        scaler: torch.cuda.amp.GradScaler,
        start_epoch: int,
        phase: str,
        use_amp: bool,
    ) -> int:
        for ep in range(n_epochs):
            epoch = start_epoch + ep
            t0 = time.time()

            train_loss, train_acc = self._train_one_epoch(
                train_loader, optimizer, loss_fn, scaler, use_amp
            )
            val_loss, val_acc, val_f1 = self._eval_one_epoch(val_loader, loss_fn)

            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                "[%s] Epoch %d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f F1=%.4f | %.1fs",
                phase,
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1,
                elapsed,
            )

            self.writer.add_scalars(
                f"{phase}/loss",
                {"train": train_loss, "val": val_loss},
                epoch,
            )
            self.writer.add_scalars(
                f"{phase}/accuracy",
                {"train": train_acc, "val": val_acc},
                epoch,
            )
            self.writer.add_scalar(f"{phase}/val_f1", val_f1, epoch)

            checkpoint.step(self.model, val_f1, epoch)
            if early_stop.step(val_f1):
                logger.info("Early stopping at epoch %d.", epoch)
                break

        return start_epoch + n_epochs

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        use_amp: bool,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = self.model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _eval_one_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # Macro F1 across all classes
        from sklearn.metrics import f1_score

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / max(total, 1), correct / max(total, 1), macro_f1
