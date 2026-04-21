"""
Cellpose segmentation wrapper for the PUMA pipeline.

``CellposeSegmentor`` wraps ``cellpose.models.CellposeModel`` or CPTransformer
(CP4) and exposes:
  * ``fine_tune()``  — fine-tune on PUMA data using the official train_seg API
  * ``predict()``    — run inference and return instance masks + centroids
  * ``save()``       — save the fine-tuned model weights

CP4 Support:
  When pretrained_model="cpsam" or path to cpsam checkpoint, uses CPTransformer
  (direct loading without full cellpose library). This is the recommended approach
  for maximum performance and flexibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from cellpose.models import CPTransformer

logger = logging.getLogger(__name__)


class CellposeSegmentor:
    """Thin wrapper around CellposeModel or CPTransformer for the PUMA segmentation stage.

    Args:
        pretrained_model: Name of a built-in Cellpose model (``"cyto3"``,
                          ``"nuclei"``…) or path to a checkpoint. Use ``"cpsam"``
                          or path to cpsam file for CP4 (recommended).
        gpu:              Use GPU if available.
        diameter:         Expected nucleus diameter in pixels.
                          ``None`` lets Cellpose estimate it automatically.
        nchan:            Number of image channels used by the model
                          (2 for standard Cellpose grayscale mode, 3 for RGB).
        use_cp_transformer: Force using CPTransformer (CP4) even with legacy model name.
    """

    def __init__(
        self,
        pretrained_model: str = "cyto3",
        gpu: bool = True,
        diameter: Optional[float] = 17.0,
        nchan: int = 2,
        use_cp_transformer: bool = False,
    ) -> None:
        from puma_seg.models.cp_transformer import CPTransformer, load_cpsam_checkpoint

        self.diameter = diameter
        self.pretrained_model = pretrained_model
        self.nchan = nchan
        self._use_cp_transformer = use_cp_transformer

        self._cp_transformer: Optional[CPTransformer] = None
        self._legacy_model: Optional[object] = None

        device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
        self._device = device

        is_cpsam = self._is_cpsam_model(pretrained_model)

        if is_cpsam or use_cp_transformer:
            logger.info("Using CPTransformer (CP4) for '%s' on %s.", pretrained_model, device)
            self._load_cp_transformer(pretrained_model, device)
        else:
            logger.info("Using legacy Cellpose model '%s' on %s.", pretrained_model, device)
            self._load_legacy_model(pretrained_model, device)

    def _is_cpsam_model(self, model_path: str) -> bool:
        """Check if the model name/path indicates CPSAM (CP4)."""
        if model_path.lower() == "cpsam":
            return True
        path = Path(model_path)
        if path.exists() and path.is_file():
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                return "W2" in ckpt
            except Exception:
                pass
        return False

    def _load_cp_transformer(self, model_path: str, device: torch.device) -> None:
        """Load CPTransformer (CP4) directly."""
        from puma_seg.models.cp_transformer import load_cpsam_checkpoint

        dtype = torch.float32

        if model_path.lower() == "cpsam":
            self._cp_transformer = load_cpsam_checkpoint(
                checkpoint_path=None, device=device, dtype=dtype
            )
        else:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"CPSAM checkpoint not found: {path}")
            self._cp_transformer = CPTransformer(dtype=dtype)
            self._cp_transformer.load_model(str(path), device=device, strict=False)
            self._cp_transformer.eval()
            self._cp_transformer.to(device)

        self._cp_transformer.eval()
        logger.info("CPTransformer loaded successfully.")

    def _load_legacy_model(self, model_path: str, device: str) -> None:
        """Load legacy Cellpose model."""
        from cellpose import models as cp_models

        self._legacy_model = cp_models.CellposeModel(
            gpu=(device == "cuda"),
            pretrained_model=model_path,
            nchan=self.nchan,
        )

    @property
    def net(self) -> torch.nn.Module:
        """Expose raw PyTorch network for advanced access."""
        if self._cp_transformer is not None:
            return self._cp_transformer
        return self._legacy_model.net

    @property
    def is_cp4(self) -> bool:
        """Return True if using CP4 (CPTransformer)."""
        return self._cp_transformer is not None

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        train_images: List[np.ndarray],
        train_labels: List[np.ndarray],
        test_images: Optional[List[np.ndarray]] = None,
        test_labels: Optional[List[np.ndarray]] = None,
        *,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 8,
        model_name: str = "puma_cellpose",
        save_path: str | Path = "./models",
        channels: Optional[List[int]] = None,
    ) -> Tuple[str, List[float], List[float]]:
        """Fine-tune the Cellpose model using the official ``train_seg`` API.

        Note: For CP4 (CPTransformer) models, the official train_seg API is NOT
        compatible. Use custom training loop instead. This method works only
        for legacy Cellpose models (cyto3, nuclei, etc.).

        Args:
            train_images: List of (H, W, 3) uint8 np.ndarray images.
            train_labels: List of (H, W) int32 instance masks.
            test_images:  Optional validation images.
            test_labels:  Optional validation instance masks.
            learning_rate: Learning rate (keep small, e.g. 1e-5 for fine-tuning).
            weight_decay:  AdamW weight decay.
            n_epochs:      Number of training epochs.
            batch_size:    Mini-batch size.
            model_name:    Filename stem for the saved checkpoint.
            save_path:     Directory to save the fine-tuned weights.
            channels:      Cellpose channels list, e.g. [0, 0] for grayscale.
                           ``None`` → [0, 0] (default grayscale mode).

        Returns:
            (model_path, train_losses, test_losses)
        """
        if self._cp_transformer is not None:
            logger.info("Fine-tuning CP4 (CPTransformer) on %d images...", len(train_images))
            result = self._cp_transformer.fine_tune(
                train_images=train_images,
                train_labels=train_labels,
                val_images=test_images,
                val_labels=test_labels,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_epochs=n_epochs,
                batch_size=batch_size,
                diameter=self.diameter or 30.0,
                device=self._device,
            )

            model_path = str(save_path / f"{model_name}")
            if result["val_losses"]:
                best_idx = np.argmin(result["val_losses"])
                logger.info(
                    "CP4 fine-tuning complete. Best val loss: %.4f at epoch %d",
                    result["val_losses"][best_idx],
                    best_idx,
                )
            else:
                logger.info(
                    "CP4 fine-tuning complete. Train loss: %.4f", result["train_losses"][-1]
                )

            model_path = str(save_path / f"{model_name}.pth")
            torch.save(self._cp_transformer.state_dict(), model_path)
            logger.info("CP4 model saved to: %s", model_path)
            return model_path, result["train_losses"], result["val_losses"]

        from cellpose import train as cp_train

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if channels is None:
            channels = [0, 0]

        logger.info(
            "Fine-tuning Cellpose: %d train | %d val | %d epochs | lr=%.1e",
            len(train_images),
            len(test_images) if test_images else 0,
            n_epochs,
            learning_rate,
        )

        model_path, train_losses, test_losses = cp_train.train_seg(
            self._legacy_model.net,
            train_data=train_images,
            train_labels=train_labels,
            test_data=test_images,
            test_labels=test_labels,
            channels=channels,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_name=model_name,
            save_path=str(save_path),
        )

        logger.info("Fine-tuning complete. Saved to: %s", model_path)
        return model_path, train_losses, test_losses

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        *,
        diameter: Optional[float] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        channels: Optional[List[int]] = None,
        tile: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Run Cellpose inference on a single image.

        Args:
            image:               (H, W, 3) or (H, W) uint8 image.
            diameter:            Override default diameter.
            flow_threshold:      Cellpose flow threshold (higher → fewer cells).
            cellprob_threshold:  Cell probability threshold.
            channels:            Cellpose channel list. ``None`` → [0, 0].
            tile:                Whether to use tiled inference for large images.

        Returns:
            instance_mask: (H, W) int32 instance mask.
            info:          Dict with centroids, bboxes, flows (for debugging).
        """
        if self._cp_transformer is not None:
            return self._predict_cp4(image, diameter, cellprob_threshold)
        else:
            return self._predict_legacy(
                image, diameter, flow_threshold, cellprob_threshold, channels, tile
            )

    def _predict_cp4(
        self,
        image: np.ndarray,
        diameter: Optional[float] = None,
        cellprob_threshold: float = 0.0,
    ) -> Tuple[np.ndarray, Dict]:
        """Inference for CP4 (CPTransformer) models."""
        from scipy import ndimage

        image = self._ensure_rgb(image)
        image_tensor = self._prepare_image_tensor(image)
        image_tensor = image_tensor.to(self._device)

        with torch.no_grad():
            self._cp_transformer.eval()
            output, _ = self._cp_transformer(image_tensor)

        output_np = output.cpu().numpy()[0]
        cellprob = output_np[1]
        flow = output_np[2:].transpose(1, 2, 0)

        masks = self._segment_cp4(cellprob, flow, cellprob_threshold)

        props = self._get_mask_properties(masks)
        centroids = (
            {
                int(inst_id): tuple(centroid)
                for inst_id, centroid in zip(props["label"], props["centroid"])
            }
            if props["label"] is not None
            else {}
        )

        info = {
            "centroids": centroids,
            "n_instances": int(masks.max()) if masks.size > 0 else 0,
            "cellprob": cellprob,
            "flow": flow,
        }
        return masks.astype(np.int32), info

    def _predict_legacy(
        self,
        image: np.ndarray,
        diameter: Optional[float],
        flow_threshold: float,
        cellprob_threshold: float,
        channels: Optional[List[int]],
        tile: bool,
    ) -> Tuple[np.ndarray, Dict]:
        """Inference for legacy Cellpose models."""
        from cellpose import utils as cp_utils

        if channels is None:
            channels = [0, 0]

        diam = diameter or self.diameter

        masks, flows, styles = self._legacy_model.eval(
            image,
            diameter=diam,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=channels,
            tile=tile,
        )

        props = cp_utils.get_masks_properties(masks)
        centroids = (
            {
                int(inst_id): tuple(centroid)
                for inst_id, centroid in zip(props["label"], props["centroid"])
            }
            if props["label"] is not None
            else {}
        )

        info = {
            "centroids": centroids,
            "flows": flows,
            "styles": styles,
            "n_instances": int(masks.max()),
        }
        return masks.astype(np.int32), info

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale to RGB if needed."""
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            return np.concatenate([image] * 3, axis=-1)
        return image

    def _prepare_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image tensor for CP4 model."""
        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0)

    def _segment_cp4(
        self, cellprob: np.ndarray, flow: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """Segment CP4 output to instance masks.

        Prefer the official Cellpose flow-based mask construction when available.
        Fallback to a lightweight connected-component + distance-peak approach.
        """
        try:
            from cellpose.dynamics import compute_masks

            masks, _, _ = compute_masks(flow[..., ::-1], cellprob, p=None)
            return masks.astype(np.int32)
        except ImportError:
            logger.debug("cellpose.dynamics unavailable, using fallback CP4 segmentation.")
            from scipy import ndimage

        cellprob_thresh = cellprob > threshold

        labels, num = ndimage.label(cellprob_thresh)
        if num == 0:
            return np.zeros_like(cellprob, dtype=np.int32)

        distances = ndimage.distance_transform_edt(cellprob_thresh)
        maxima = self._find_local_maxima(distances)

        masks = np.zeros_like(cellprob, dtype=np.int32)
        for i, (y, x) in enumerate(maxima, start=1):
            masks[labels == labels[y, x]] = i

        return masks

    def _find_local_maxima(self, distances: np.ndarray, min_distance: int = 10) -> list:
        """Find local maxima in distance map."""
        from scipy.ndimage import maximum_filter

        max_filtered = maximum_filter(distances, size=min_distance)
        peaks = (distances == max_filtered) & (distances > 0)
        coords = np.argwhere(peaks)
        return coords.tolist()

    def _get_mask_properties(self, masks: np.ndarray) -> Dict:
        """Get mask properties without cellpose utils."""
        from scipy import ndimage

        if masks.max() == 0:
            return {"label": None, "centroid": None}

        labels = []
        centroids = []

        for i in range(1, int(masks.max()) + 1):
            mask = masks == i
            if mask.any():
                coords = np.argwhere(mask)
                labels.append(i)
                centroid = coords.mean(axis=0)
                centroids.append((float(centroid[0]), float(centroid[1])))

        return {"label": labels, "centroid": centroids}

    def predict_batch(
        self,
        images: List[np.ndarray],
        **kwargs,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """Predict on a list of images."""
        return [self.predict(img, **kwargs) for img in images]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the current model weights (``self.net``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._cp_transformer is not None:
            torch.save(self._cp_transformer.state_dict(), str(path))
            logger.info("CPTransformer weights saved to: %s", path)
        else:
            torch.save(self._legacy_model.net.state_dict(), str(path))
            logger.info("Cellpose net weights saved to: %s", path)

    def load_weights(self, path: str | Path) -> None:
        """Load weights into ``self.net`` from a file saved with ``save()``."""
        path = Path(path)
        state_dict = torch.load(str(path), map_location="cpu")

        if "W2" in state_dict:
            if self._cp_transformer is None:
                self._load_cp_transformer(str(path), self._device)
            else:
                self._cp_transformer.load_state_dict(state_dict)
            logger.info("Loaded CPTransformer weights from: %s", path)
        else:
            self._legacy_model.net.load_state_dict(state_dict)
            logger.info("Loaded legacy Cellpose weights from: %s", path)
