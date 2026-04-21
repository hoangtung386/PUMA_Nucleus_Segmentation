"""
CP-SAM Transformer (CP4) - Direct loading from Cellpose checkpoint.

This module provides the Transformer class that replicates the Cellpose CP4 architecture,
allowing direct loading of the 'cpsam' checkpoint without the full cellpose library.

Architecture details:
- Based on ViT-L backbone from SAM (Segment Anything Model)
- Modified patch size (default 8x8)
- Custom output head with W2 matrix for pixel space transformation
- Global attention (window_size=0) in all transformer blocks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cp4_dataset import CP4Dataset, CP4Loss

logger = logging.getLogger(__name__)

try:
    from segment_anything import sam_model_registry

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("segment_anything not installed. CPTransformer will not work.")


class CPTransformer(nn.Module):
    """CP-SAM Transformer (CP4) model for cell segmentation.

    This model is based on the SAM ViT-L backbone with modifications:
    - Custom patch size (default 8x8 instead of 16x16)
    - Adjusted position embeddings for different token sizes
    - W2 matrix for token-to-pixel space transformation
    - Global attention in all transformer blocks

    Args:
        backbone: SAM backbone architecture ('vit_l' for CP4).
        ps: Patch size for the input embedding (default 8).
        nout: Number of output channels (default 3).
        bsize: Base size for position embedding adjustment (default 256).
        rdrop: Dropout fraction for transformer layers during training (default 0.4).
        checkpoint: Path to SAM checkpoint (optional, for initialization).
        dtype: Data type for model weights.
    """

    def __init__(
        self,
        backbone: str = "vit_l",
        ps: int = 8,
        nout: int = 3,
        bsize: int = 256,
        rdrop: float = 0.4,
        checkpoint: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if not SAM_AVAILABLE:
            raise ImportError(
                "segment_anything is required. Install with: pip install segment-anything"
            )

        self.encoder = sam_model_registry[backbone](checkpoint).image_encoder
        w = self.encoder.patch_embed.proj.weight.detach()
        nchan = w.shape[0]

        self.ps = ps
        self.encoder.patch_embed.proj = nn.Conv2d(3, nchan, stride=ps, kernel_size=ps)
        self.encoder.patch_embed.proj.weight.data = w[:, :, :: 16 // ps, :: 16 // ps]

        target_h = 256 // ps
        target_w = 256 // ps
        orig_shape = self.encoder.pos_embed.shape
        orig_h = orig_shape[1]
        orig_w = orig_shape[2]

        if target_h != orig_h or target_w != orig_w:
            pos_embed_2d = self.encoder.pos_embed.permute(0, 3, 1, 2)
            pos_embed_resized = F.interpolate(
                pos_embed_2d,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            self.encoder.pos_embed = nn.Parameter(
                pos_embed_resized.permute(0, 2, 3, 1),
                requires_grad=True,
            )

        self.nout = nout
        self.out = nn.Conv2d(256, self.nout * ps**2, kernel_size=1)

        self.W2 = nn.Parameter(
            torch.eye(self.nout * ps**2).reshape(self.nout * ps**2, self.nout, ps, ps),
            requires_grad=False,
        )

        self.rdrop = rdrop
        self.diam_labels = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.diam_mean = nn.Parameter(torch.tensor([30.0]), requires_grad=False)

        for blk in self.encoder.blocks:
            blk.window_size = 0

        self._dtype = dtype
        if dtype != torch.float32:
            self.to(dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the model."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """Set the data type of the model."""
        if self._dtype != value:
            self.to(value)
            self._dtype = value

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the CPTransformer.

        Args:
            x: (B, 3, H, W) input图像，已normalize

        Returns:
            output: (B, nout, H, W) segmentation output
            style: (B, 256) style features for cellpose compatibility
        """
        x = x.to(self.dtype)
        x = self.encoder.patch_embed(x)

        if self.encoder.pos_embed is not None:
            _, h, w, c = x.shape
            pos_embed = self.encoder.pos_embed

            if pos_embed.shape[1] != h or pos_embed.shape[2] != w:
                pos_embed_3d = pos_embed.permute(0, 3, 1, 2)
                pos_embed_resized = F.interpolate(
                    pos_embed_3d,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                pos_embed = pos_embed_resized.permute(0, 2, 3, 1)

            x = x + pos_embed

        if self.training and self.rdrop > 0:
            nlay = len(self.encoder.blocks)
            rdrop = (
                torch.rand((len(x), nlay), device=x.device)
                < torch.linspace(0, self.rdrop, nlay, device=x.device)
            ).to(x.dtype)
            for i, blk in enumerate(self.encoder.blocks):
                mask = rdrop[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x * mask + blk(x) * (1 - mask)
        else:
            for blk in self.encoder.blocks:
                x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))

        x1 = self.out(x)
        x1 = F.conv_transpose2d(x1, self.W2, stride=self.ps, padding=0)

        style = torch.zeros((x.shape[0], 256), device=x.device)

        return x1, style

    def load_model(self, PATH: str | Path, device: torch.device, strict: bool = False) -> None:
        """Load CP4 (cpsam) checkpoint.

        Args:
            PATH: Path to the cpsam checkpoint file.
            device: Device to load the model on.
            strict: Whether to strictly enforce state_dict key matching.
        """
        state_dict = torch.load(PATH, map_location=device, weights_only=True)

        w2_data = state_dict.get("W2", None)
        if w2_data is None:
            raise ValueError(
                "This model does not appear to be a CP4 model. "
                "CP3 models are not compatible with CP4."
            )

        keys = list(state_dict.keys())
        if keys[0][:7] == "module.":
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict=strict)
        else:
            self.load_state_dict(state_dict, strict=strict)

        if self.dtype != torch.float32:
            self.to(self.dtype)

        logger.info("Loaded CP4 checkpoint from: %s", PATH)

    def save_model(self, filename: str | Path) -> None:
        """Save model to a file.

        Args:
            filename: Path where the model will be saved.
        """
        torch.save(self.state_dict(), filename)

    def get_flow(self, image: torch.Tensor, diameter: float = 30.0) -> torch.Tensor:
        """Compute flow field from input image.

        This is a simplified inference that returns the model's raw output.
        For full cellpose-style inference, use the CellposeSegmentor class.

        Args:
            image: (B, 3, H, W) input tensor.
            diameter: Expected cell diameter (used for normalization).

        Returns:
            flow: (B, 3, H, W) flow field prediction.
        """
        with torch.no_grad():
            self.eval()
            flow, _ = self.forward(image)
        return flow

    def fine_tune(
        self,
        train_images: List[np.ndarray],
        train_labels: List[np.ndarray],
        val_images: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[np.ndarray]] = None,
        *,
        diameter: float = 30.0,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 8,
        save_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[float]]:
        """Fine-tune CP4 on custom data.

        Args:
            train_images: List of (H, W, 3) uint8 images.
            train_labels: List of (H, W) int32 instance masks.
            val_images: Optional validation images.
            val_labels: Optional validation masks.
            diameter: Expected cell diameter.
            learning_rate: Learning rate.
            weight_decay: AdamW weight decay.
            n_epochs: Number of epochs.
            batch_size: Batch size.
            save_path: Path to save best checkpoint.
            device: Device to train on.

        Returns:
            dict with train_losses, val_losses.
        """
        if device is None:
            device = next(self.parameters()).device

        train_ds = CP4Dataset(train_images, train_labels, diameter, augment=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        if val_images is not None and val_labels is not None:
            val_ds = CP4Dataset(val_images, val_labels, diameter, augment=False)
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )

        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=learning_rate * 0.01
        )
        loss_fn = CP4Loss()

        best_val_loss = float("inf")
        train_losses, val_losses = [], []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for img, y_flow, x_flow, cellprob in train_loader:
                img = img.to(device).to(self.dtype)
                y_flow = y_flow.to(device).to(self.dtype)
                x_flow = x_flow.to(device).to(self.dtype)
                cellprob = cellprob.to(device).to(self.dtype)

                optimizer.zero_grad()
                pred, _ = self.forward(img)
                loss = loss_fn(pred, y_flow, x_flow, cellprob)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            val_loss = None
            if val_loader is not None:
                self.eval()
                val_epoch_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for img, y_flow, x_flow, cellprob in val_loader:
                        img = img.to(device).to(self.dtype)
                        y_flow = y_flow.to(device).to(self.dtype)
                        x_flow = x_flow.to(device).to(self.dtype)
                        cellprob = cellprob.to(device).to(self.dtype)

                        pred, _ = self.forward(img)
                        loss = loss_fn(pred, y_flow, x_flow, cellprob)

                        val_epoch_loss += loss.item()
                        val_batches += 1

                val_loss = val_epoch_loss / max(val_batches, 1)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        self.save_model(save_path)

                self.train()

            scheduler.step()

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%s",
                epoch + 1,
                n_epochs,
                train_loss,
                f"{val_loss:.4f}" if val_loss else "N/A",
            )

        return {"train_losses": train_losses, "val_losses": val_losses}


def load_cpsam_checkpoint(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> CPTransformer:
    """Load CPTransformer with CPSAM checkpoint.

    Args:
        checkpoint_path: Path to cpsam checkpoint. If None, will download from HuggingFace.
        device: Device to load on. Defaults to cuda if available.
        dtype: Data type for the model.

    Returns:
        Loaded CPTransformer model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CPTransformer(dtype=dtype)

    if checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id="mouseland/cellpose-sam",
                filename="cpsam",
                repo_type="model",
            )
            logger.info("Downloaded cpsam from HuggingFace: %s", checkpoint_path)
        except ImportError:
            raise ImportError(
                "huggingface_hub required for automatic download. "
                "Install with: pip install huggingface-hub\n"
                "Or provide checkpoint_path manually."
            )

    model.load_model(checkpoint_path, device=device, strict=False)
    model.eval()
    model.to(device)

    logger.info("CPTransformer loaded successfully on %s", device)
    return model
