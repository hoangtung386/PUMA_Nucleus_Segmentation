"""
Simple E6 inference script with colored visualization masks.

Purpose:
- Load one trained E6 checkpoint.
- Load one image path.
- Run model inference only.
- Save raw prediction masks for evaluation/submission.
- Save colored PNG masks for visual checking.

No fold logic. No validation split. No ground-truth loading.

Put this file in the project folder, next to the `encoder/` directory.
Then edit only the USER SETTINGS section below.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import tifffile
except Exception:
    tifffile = None

from encoder.config import ENCODER_RUNS
from encoder.data import IMAGENET_MEAN, IMAGENET_STD
from encoder.engine import _initialize_lazy_modules
from encoder.models import PumaEncoderProbe


# ==========================================================
# USER SETTINGS: edit these paths only
# ==========================================================

# Your trained E6 checkpoint.
checkpoint_path = Path(
    "/content/drive/MyDrive/Research/PUMA/checkpoints_encoder/E6_UNIv2_lora/fold_1/best_model.pth"
)

# The image you want to run inference on.
image_path = Path(
    "/content/drive/MyDrive/Research/PUMA/Dataset/01_training_dataset_tif_ROIs/training_set_metastatic_roi_001.tif"
)

# Where to save outputs.
output_dir = Path(
    "/content/drive/MyDrive/Research/PUMA/e6_inference_outputs"
)

# E6 was trained with 1024x1024 ROI input in your current code.
image_size = 1024

# Save raw ID masks. Keep this True for useful inference output.
save_raw_masks = True

# Save colored PNG masks for visual inspection.
save_colored_previews = True

# Save a side-by-side overview PNG: original image + tissue + nuclei foreground + nuclei class.
save_overview_png = True


# ==========================================================
# Label meaning for your current E6 code
# ==========================================================

# Tissue prediction is already in official PUMA Track 2 IDs:
# 0 = background
# 1 = stroma
# 2 = blood vessel
# 3 = tumor
# 4 = epidermis / epithelium
# 5 = necrosis
# Therefore, no tissue remapping is needed here.

TISSUE_ID_TO_NAME = {
    0: "background",
    1: "stroma",
    2: "blood_vessel",
    3: "tumor",
    4: "epidermis",
    5: "necrosis",
}

NUCLEI_ID_TO_NAME = {
    0: "tumor",
    1: "lymphocyte",
    2: "plasma_cell",
    3: "histiocyte",
    4: "melanophage",
    5: "neutrophil",
    6: "stroma",
    7: "epithelium",
    8: "endothelium",
    9: "apoptosis",
}

# RGB colors for official PUMA tissue IDs.
TISSUE_COLORS = {
    0: (0, 0, 0),          # background - black
    1: (0, 180, 0),        # stroma - green
    2: (0, 180, 255),      # blood vessel - cyan
    3: (255, 0, 0),        # tumor - red
    4: (255, 200, 0),      # epidermis / epithelium - yellow/orange
    5: (120, 0, 180),      # necrosis - purple
}

# RGB colors for nuclei class IDs.
# 255 is used only for visualization/background after masking with nuclei foreground.
NUCLEI_COLORS_WITH_BG = {
    255: (0, 0, 0),        # background - black
    0: (255, 0, 0),        # tumor - red
    1: (0, 0, 255),        # lymphocyte - blue
    2: (255, 0, 255),      # plasma cell - magenta
    3: (255, 140, 0),      # histiocyte - orange
    4: (120, 70, 20),      # melanophage - brown
    5: (0, 255, 255),      # neutrophil - cyan
    6: (0, 180, 0),        # stromal - green
    7: (255, 220, 0),      # epithelium - yellow
    8: (0, 150, 150),      # endothelium - teal
    9: (180, 180, 180),    # apoptosis - gray
}

NUCLEI_FG_COLORS = {
    0: (0, 0, 0),          # background - black
    1: (255, 255, 255),    # nuclei foreground - white
}


def read_rgb_image(path: Path) -> np.ndarray:
    """Read tif/png/jpg image as uint8 RGB numpy array [H, W, 3]."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        img = tifffile.imread(str(path))
    else:
        img = np.array(Image.open(path))

    # Handle channel layout and grayscale/RGBA cases.
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[0] in {3, 4} and img.shape[-1] not in {3, 4}:
        # Convert [C, H, W] to [H, W, C] if needed.
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim > 3:
        # If a TIFF has extra dimensions, use the first plane.
        img = np.squeeze(img)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[0] in {3, 4} and img.shape[-1] not in {3, 4}:
            img = np.transpose(img, (1, 2, 0))

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape after loading: {img.shape}")

    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape: {img.shape}")

    # Convert to uint8 if needed.
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        max_value = float(np.nanmax(img)) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0
        elif max_value > 255.0:
            img = img / max_value * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def resize_if_needed(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image to model input size if needed."""
    h, w = img.shape[:2]
    if h == size and w == size:
        return img

    print(f"Input image is {h}x{w}. Resizing to {size}x{size} for E6 model.")
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB [H, W, 3] to normalized tensor [1, 3, H, W]."""
    x = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Save mask as tif if tifffile exists, otherwise save as png."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        tifffile.imwrite(str(path), mask.astype(np.uint8))
    else:
        Image.fromarray(mask.astype(np.uint8)).save(path)


def colorize_mask(mask: np.ndarray, color_dict: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """
    Convert a 2D label mask into an RGB image.

    Unknown labels are colored black by default.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape: {mask.shape}")

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_dict.items():
        rgb[mask == class_id] = color

    return rgb


def save_color_mask(mask: np.ndarray, color_dict: dict[int, tuple[int, int, int]], save_path: Path) -> None:
    """Save a colored RGB PNG from a 2D label mask."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = colorize_mask(mask, color_dict)
    Image.fromarray(rgb).save(save_path)


def save_overlay(
    image_rgb: np.ndarray,
    color_mask_rgb: np.ndarray,
    save_path: Path,
    alpha: float = 0.45,
) -> None:
    """Save a simple overlay of color mask on top of original RGB image."""
    if image_rgb.shape[:2] != color_mask_rgb.shape[:2]:
        raise ValueError(
            f"Image and mask sizes do not match: {image_rgb.shape} vs {color_mask_rgb.shape}"
        )

    image_f = image_rgb.astype(np.float32)
    mask_f = color_mask_rgb.astype(np.float32)
    overlay = (1.0 - alpha) * image_f + alpha * mask_f
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(save_path)


def make_overview_png(
    image_rgb: np.ndarray,
    tissue_rgb: np.ndarray,
    nuclei_fg_rgb: np.ndarray,
    nuclei_class_rgb: np.ndarray,
    save_path: Path,
) -> None:
    """Save one side-by-side PNG for quick visual checking."""
    h, w = image_rgb.shape[:2]
    panels = [image_rgb, tissue_rgb, nuclei_fg_rgb, nuclei_class_rgb]
    canvas = np.zeros((h, w * len(panels), 3), dtype=np.uint8)

    for i, panel in enumerate(panels):
        canvas[:, i * w : (i + 1) * w] = panel

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(save_path)


def print_class_counts(name: str, mask: np.ndarray, id_to_name: dict[int, str], ignore_id: int | None = None) -> None:
    """Print pixel counts for a predicted mask."""
    print(f"\n{name} pixel counts:")
    unique_ids, counts = np.unique(mask, return_counts=True)
    for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
        if ignore_id is not None and class_id == ignore_id:
            class_name = "background_or_non_nuclei"
        else:
            class_name = id_to_name.get(int(class_id), "unknown")
        print(f"  {class_id}: {class_name}: {count}")


def load_e6_model(checkpoint_path: Path, device: torch.device) -> PumaEncoderProbe:
    """Build E6 model, initialize lazy layers, and load checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # E6 config: UNIv2 LoRA. Use pretrained=False because the checkpoint already
    # contains the trained weights. This avoids unnecessary pretrained download.
    cfg = replace(ENCODER_RUNS["E6"], pretrained=False, image_size=image_size)

    model = PumaEncoderProbe(cfg).to(device)

    use_amp = bool(cfg.amp and device.type == "cuda")
    _initialize_lazy_modules(model=model, cfg=cfg, device=device, use_amp=use_amp)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


def run_inference() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_e6_model(checkpoint_path, device)

    img = read_rgb_image(image_path)
    original_h, original_w = img.shape[:2]
    img_for_model = resize_if_needed(img, image_size)
    x = preprocess_image(img_for_model).to(device)

    use_amp = device.type == "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(x)

    tissue_pred = outputs["tissue"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    nuclei_fg_pred = outputs["nuclei_fg"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    nuclei_class_pred = outputs["nuclei_class"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Keep nuclei class only where the model predicts nuclei foreground.
    # Background pixels are set to 255 because nuclei_class has no real background class.
    nuclei_class_masked = np.where(nuclei_fg_pred > 0, nuclei_class_pred, 255).astype(np.uint8)

    # If the original image was not 1024x1024, resize predictions back to original size.
    if (original_h, original_w) != (image_size, image_size):
        tissue_pred = np.array(
            Image.fromarray(tissue_pred).resize((original_w, original_h), resample=Image.NEAREST),
            dtype=np.uint8,
        )
        nuclei_fg_pred = np.array(
            Image.fromarray(nuclei_fg_pred).resize((original_w, original_h), resample=Image.NEAREST),
            dtype=np.uint8,
        )
        nuclei_class_masked = np.array(
            Image.fromarray(nuclei_class_masked).resize((original_w, original_h), resample=Image.NEAREST),
            dtype=np.uint8,
        )

    stem = image_path.stem

    if save_raw_masks:
        # Save numpy arrays.
        np.save(output_dir / f"{stem}_tissue_puma_ids.npy", tissue_pred)
        np.save(output_dir / f"{stem}_nuclei_fg.npy", nuclei_fg_pred)
        np.save(output_dir / f"{stem}_nuclei_class_masked.npy", nuclei_class_masked)

        # Save raw ID masks. These are the important files for evaluation/submission-style use.
        save_mask(output_dir / f"{stem}_tissue_puma_ids.tif", tissue_pred)
        save_mask(output_dir / f"{stem}_nuclei_fg.tif", nuclei_fg_pred)
        save_mask(output_dir / f"{stem}_nuclei_class_masked.tif", nuclei_class_masked)

    if save_colored_previews or save_overview_png:
        tissue_rgb = colorize_mask(tissue_pred, TISSUE_COLORS)
        nuclei_fg_rgb = colorize_mask(nuclei_fg_pred, NUCLEI_FG_COLORS)
        nuclei_class_rgb = colorize_mask(nuclei_class_masked, NUCLEI_COLORS_WITH_BG)

        if save_colored_previews:
            Image.fromarray(tissue_rgb).save(output_dir / f"{stem}_tissue_colored.png")
            Image.fromarray(nuclei_fg_rgb).save(output_dir / f"{stem}_nuclei_fg_colored.png")
            Image.fromarray(nuclei_class_rgb).save(output_dir / f"{stem}_nuclei_class_colored.png")

            save_overlay(
                image_rgb=img,
                color_mask_rgb=tissue_rgb,
                save_path=output_dir / f"{stem}_tissue_overlay.png",
                alpha=0.45,
            )
            save_overlay(
                image_rgb=img,
                color_mask_rgb=nuclei_class_rgb,
                save_path=output_dir / f"{stem}_nuclei_class_overlay.png",
                alpha=0.45,
            )

        if save_overview_png:
            make_overview_png(
                image_rgb=img,
                tissue_rgb=tissue_rgb,
                nuclei_fg_rgb=nuclei_fg_rgb,
                nuclei_class_rgb=nuclei_class_rgb,
                save_path=output_dir / f"{stem}_overview.png",
            )

    print_class_counts("Tissue", tissue_pred, TISSUE_ID_TO_NAME)
    print_class_counts("Nuclei foreground", nuclei_fg_pred, {0: "background", 1: "nuclei"})
    print_class_counts("Nuclei class masked", nuclei_class_masked, NUCLEI_ID_TO_NAME, ignore_id=255)

    print("\nInference finished.")
    print(f"Saved outputs to: {output_dir}")
    print("\nTissue output uses official PUMA IDs:")
    for k, v in TISSUE_ID_TO_NAME.items():
        print(f"  {k}: {v}")

    print("\nColored files are only for visualization. Raw .tif/.npy masks keep the class IDs.")


if __name__ == "__main__":
    run_inference()
