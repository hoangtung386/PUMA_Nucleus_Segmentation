"""
Visualisation utilities for PUMA nucleus segmentation results.

Functions:
  overlay_instances      — draw coloured instance mask over the H&E image
  color_code_classes     — colour each nucleus by type
  plot_sample            — show a single image + mask pair
  plot_predictions_vs_gt — side-by-side comparison
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

# ─── Colour palettes ──────────────────────────────────────────────────────────

# Track 1: background, tumor, lymphocyte, other
COLORS_T1: List[Tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),    # 0 background — black
    (0.9, 0.2, 0.2),    # 1 tumor       — red
    (0.2, 0.4, 0.9),    # 2 lymphocyte  — blue
    (0.2, 0.75, 0.3),   # 3 other       — green
]

# Track 2: background + 10 classes
COLORS_T2: List[Tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),   # background
    (0.9, 0.2, 0.2),   # tumor
    (0.2, 0.4, 0.9),   # lymphocyte
    (0.6, 0.2, 0.8),   # plasma cell
    (1.0, 0.6, 0.1),   # histiocyte
    (0.5, 0.3, 0.1),   # melanophage
    (0.9, 0.9, 0.2),   # neutrophil
    (0.3, 0.8, 0.8),   # stroma
    (0.0, 0.6, 0.5),   # endothelium
    (0.9, 0.5, 0.7),   # epithelium
    (0.5, 0.5, 0.5),   # apoptosis
]


def color_code_classes(
    class_mask: np.ndarray,
    track: int = 1,
    alpha: float = 0.5,
) -> np.ndarray:
    """Convert a per-pixel class mask to an RGBA colour image.

    Args:
        class_mask: (H, W) uint8 array with class IDs.
        track:      1 or 2 (selects colour palette).
        alpha:      Opacity of foreground classes.

    Returns:
        (H, W, 4) float32 RGBA image in [0, 1].
    """
    palette = COLORS_T1 if track == 1 else COLORS_T2
    H, W = class_mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    for cls_id, colour in enumerate(palette):
        mask = class_mask == cls_id
        a = 0.0 if cls_id == 0 else alpha  # background is transparent
        rgba[mask] = (*colour, a)

    return rgba


def overlay_instances(
    image: np.ndarray,
    instance_mask: np.ndarray,
    class_mask: Optional[np.ndarray] = None,
    track: int = 1,
    alpha: float = 0.4,
    contour_width: int = 2,
) -> np.ndarray:
    """Overlay nucleus instances on the original H&E image.

    Args:
        image:         (H, W, 3) uint8 RGB image.
        instance_mask: (H, W) int32 instance mask.
        class_mask:    (H, W) uint8 class mask. If provided, instances are
                       coloured by class; otherwise random per-instance colour.
        track:         1 or 2.
        alpha:         Blend factor of the mask overlay.
        contour_width: Width of nucleus boundary contours (px).

    Returns:
        (H, W, 3) uint8 blended image.
    """
    import cv2

    image_f = image.astype(np.float32) / 255.0

    if class_mask is not None:
        # Colour by class
        colour_overlay = color_code_classes(class_mask, track=track, alpha=1.0)
        colour_rgb = colour_overlay[..., :3]
        fg = class_mask > 0
    else:
        # Random per-instance colour using skimage
        colour_rgb = label2rgb(instance_mask, bg_label=0, kind="overlay")
        fg = instance_mask > 0

    blended = image_f.copy()
    blended[fg] = (1 - alpha) * image_f[fg] + alpha * colour_rgb[fg]
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    # Draw contours
    if contour_width > 0:
        for inst_id in np.unique(instance_mask):
            if inst_id == 0:
                continue
            binary = (instance_mask == inst_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(blended, contours, -1, (255, 255, 255), contour_width)

    return blended


def plot_sample(
    image: np.ndarray,
    instance_mask: np.ndarray,
    class_mask: Optional[np.ndarray] = None,
    track: int = 1,
    title: str = "",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot a single H&E image alongside its nucleus overlay."""
    from puma_seg.data.geojson_parser import CLASS_NAMES_T1, CLASS_NAMES_T2

    palette = COLORS_T1 if track == 1 else COLORS_T2
    class_names = CLASS_NAMES_T1 if track == 1 else CLASS_NAMES_T2

    overlay = overlay_instances(image, instance_mask, class_mask, track=track)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image)
    axes[0].set_title("Original H&E")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Nucleus overlay  (n={instance_mask.max()})")
    axes[1].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=palette[i], label=class_names[i])
        for i in range(1, len(class_names))
    ]
    axes[1].legend(handles=patches, loc="lower right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_predictions_vs_gt(
    image: np.ndarray,
    pred_instance_mask: np.ndarray,
    gt_instance_mask: np.ndarray,
    pred_class_mask: Optional[np.ndarray] = None,
    gt_class_mask: Optional[np.ndarray] = None,
    track: int = 1,
    title: str = "",
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side: original image | GT overlay | prediction overlay."""
    pred_overlay = overlay_instances(image, pred_instance_mask, pred_class_mask, track)
    gt_overlay = overlay_instances(image, gt_instance_mask, gt_class_mask, track)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, img, t in zip(
        axes,
        [image, gt_overlay, pred_overlay],
        ["Original H&E", f"Ground Truth  (n={gt_instance_mask.max()})",
         f"Prediction  (n={pred_instance_mask.max()})"],
    ):
        ax.imshow(img)
        ax.set_title(t)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Curve",
    save_path: Optional[str] = None,
) -> None:
    """Plot training (and optional validation) loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", color="#e63946")
    if val_losses:
        ax.plot(epochs, val_losses, label="Val", color="#457b9d", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
