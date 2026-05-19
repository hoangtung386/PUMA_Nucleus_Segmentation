"""Cellpose flow generation and HV map computation for preprocessing and inference."""

from typing import Optional

import cv2
import numpy as np
import torch

from training.logging_utils import logger


def compute_hv_map(inst_mask: np.ndarray) -> np.ndarray:
    """HoVer-Net style horizontal/vertical maps, shape [2, H, W]."""
    h_map = np.zeros_like(inst_mask, dtype=np.float32)
    v_map = np.zeros_like(inst_mask, dtype=np.float32)

    for inst_id in np.unique(inst_mask):
        if inst_id == 0:
            continue
        ys, xs = np.where(inst_mask == inst_id)
        if len(xs) == 0:
            continue
        x_center = float(xs.mean())
        y_center = float(ys.mean())
        x_radius = max((float(xs.max()) - float(xs.min())) / 2.0, 1.0)
        y_radius = max((float(ys.max()) - float(ys.min())) / 2.0, 1.0)
        h_map[ys, xs] = np.clip((xs - x_center) / (x_radius + 1e-8), -1.0, 1.0)
        v_map[ys, xs] = np.clip((ys - y_center) / (y_radius + 1e-8), -1.0, 1.0)

    return np.stack([h_map, v_map], axis=0).astype(np.float16)


def compute_hv_map_torch(inst_map: torch.Tensor) -> torch.Tensor:
    """PyTorch version of HV distance map for inference."""
    return torch.from_numpy(compute_hv_map(inst_map.cpu().numpy()))


class CellposeFlowGenerator:
    """Shared Cellpose flow generator for preprocessing (NumPy) and inference (PyTorch).

    Preprocessing usage: CellposeFlowGenerator(enabled=True, model_type="cyto3")
    Inference usage: CellposeFlowGenerator(model_type="nuclei", mode="auto", device=device)
    """

    def __init__(self, enabled: bool = True, model_type: str = "nuclei", mode: str = "auto", device: Optional[torch.device] = None):
        self.model_type = model_type
        self.device = device or torch.device("cpu")
        self.model = None

        if not enabled or mode == "zero":
            logger.info("Cellpose flow generation disabled (enabled=%s, mode=%s)", enabled, mode)
            return

        try:
            from cellpose import models as cellpose_models
            self.model = cellpose_models.CellposeModel(gpu=(self.device.type == "cuda"), model_type=model_type)
            logger.info("Cellpose loaded model_type=%s gpu=%s", model_type, self.device.type == "cuda")
        except Exception as exc:
            if mode == "generate":
                raise RuntimeError(f"Cellpose could not be loaded: {exc}") from exc
            logger.warning("Cellpose could not be loaded (%s). Using zero flow.", exc)

    def _run_cellpose(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run Cellpose model and return flow array [2, H, W]."""
        h, w = image_rgb.shape[:2]
        result = self.model.eval(
            image_rgb,
            diameter=None,
            channels=[0, 0],
            flow_threshold=None,
            cellprob_threshold=0.0,
        )
        if len(result) == 4:
            _, flows, _, _ = result
        else:
            _, flows, _ = result
        flow = flows[1] if isinstance(flows, list) and len(flows) > 1 else flows
        flow = np.asarray(flow)
        if flow.ndim == 3 and flow.shape[0] >= 2:
            flow = flow[:2]
        elif flow.ndim == 3 and flow.shape[-1] >= 2:
            flow = flow[..., :2].transpose(2, 0, 1)
        else:
            raise RuntimeError(f"Unexpected Cellpose flow shape: {flow.shape}")
        if flow.shape[1] != h or flow.shape[2] != w:
            flow = np.stack([
                cv2.resize(flow[0], (w, h), interpolation=cv2.INTER_LINEAR),
                cv2.resize(flow[1], (w, h), interpolation=cv2.INTER_LINEAR),
            ], axis=0)
        return flow

    def make_flow(self, image_rgb: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray | torch.Tensor:
        """Generate flow.

        Args:
            image_rgb: HWC uint8 image.
            device: If provided, returns [1, 2, H, W] float32 tensor on device.
                    If None, returns [2, H, W] float16 numpy array.
        """
        h, w = image_rgb.shape[:2]
        if self.model is None:
            flow = np.zeros((2, h, w), dtype=np.float16)
        else:
            try:
                flow = self._run_cellpose(image_rgb).astype(np.float32)
            except Exception as exc:
                logger.warning("Cellpose flow failed (%s). Using zero flow for this tile.", exc)
                flow = np.zeros((2, h, w), dtype=np.float16 if device is None else np.float32)

        if device is not None:
            return torch.from_numpy(flow).unsqueeze(0).float().to(device)
        return flow.astype(np.float16)
