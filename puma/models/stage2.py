from __future__ import annotations

import contextlib
import gc
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from puma.config import (
    PFM_SPECS,
    PUMA_CLASS_NAMES,
    REJECT_CLASS_ID,
    STAGE2_GEOMETRY_DIM,
    Stage2ModelConfig,
)


class LoRALinear(nn.Module):
    """Low-rank residual adapter for an existing linear projection."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base = base
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B
        ) * self.scale
        return self.base(x) + residual


class LoRAQKV(nn.Module):
    """LoRA for a combined QKV projection, updating query and value only."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError(
                "Combined QKV projection must have out_features divisible by three."
            )
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base = base
        self.dimension = base.out_features // 3
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B_qv = nn.Parameter(torch.zeros(2 * self.dimension, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        qv = F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B_qv
        ) * self.scale
        query_delta, value_delta = qv.split(self.dimension, dim=-1)
        key_delta = torch.zeros_like(query_delta)
        return base_output + torch.cat(
            [query_delta, key_delta, value_delta], dim=-1
        )


def inject_lora_last_blocks(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    last_n_blocks: int,
) -> int:
    """Inject LoRA into UNI2-h attention projections in the final blocks."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("UNI2-h encoder does not expose a transformer 'blocks' list.")
    blocks = list(blocks)
    if not blocks:
        raise RuntimeError("UNI2-h encoder contains no transformer blocks.")
    if last_n_blocks <= 0:
        raise ValueError("lora_last_blocks must be positive.")

    injected = 0
    for block in blocks[-min(last_n_blocks, len(blocks)) :]:
        attention = getattr(block, "attn", None)
        if attention is None:
            continue
        combined = getattr(attention, "qkv", None)
        if isinstance(combined, nn.Linear):
            attention.qkv = LoRAQKV(
                combined, rank=rank, alpha=alpha, dropout=dropout
            )
            injected += 1
            continue
        for name in ("q_proj", "v_proj", "q", "v"):
            layer = getattr(attention, name, None)
            if isinstance(layer, nn.Linear):
                setattr(
                    attention,
                    name,
                    LoRALinear(
                        layer, rank=rank, alpha=alpha, dropout=dropout
                    ),
                )
                injected += 1
    if injected == 0:
        raise RuntimeError(
            "No UNI2-h Q/V attention projections were found for LoRA injection."
        )
    return injected


_STAGE2_PFM_KEYS: tuple[str, ...] = ("uni2_h",)
UNI2_CHECKPOINT_FILENAME = "uni2_h_model.bin"
_MIN_VALID_CHECKPOINT_BYTES = 1_000_000
_RUNTIME_CHECKPOINT_PATH: Path | None = None


def uni2_checkpoint_path(project_root: Path | str) -> Path:
    """Return the single persistent UNI2-h binary used by every later run."""
    root = Path(project_root).expanduser().resolve()
    return root / "PUMA_pretrained_checkpoints" / "UNI2-h" / UNI2_CHECKPOINT_FILENAME


def _project_root_from_environment() -> Path:
    configured = os.environ.get("PUMA_PROJECT_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _is_valid_checkpoint_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_VALID_CHECKPOINT_BYTES


def _uni2_architecture_kwargs() -> dict[str, Any]:
    """Exact UNI2-h constructor arguments from the locked Stage-2 design."""
    from timm.layers import SwiGLUPacked

    return {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }


def _create_uni2_architecture() -> nn.Module:
    """Instantiate UNI2-h without contacting Hugging Face."""
    from timm.models.vision_transformer import VisionTransformer

    return VisionTransformer(**_uni2_architecture_kwargs())


def _download_uni2_hub_model(hf_token: str | None) -> nn.Module:
    """One-time gated load used only to create the persistent project binary."""
    import timm

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    spec = PFM_SPECS["uni2_h"]
    return timm.create_model(
        spec["hf_model"], pretrained=True, **_uni2_architecture_kwargs()
    )


def _atomic_save_state_dict(model: nn.Module, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        # Save a plain state_dict so later runs need only torch + timm and never need
        # to parse the Hugging Face snapshot again.
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _has_existing_huggingface_weights(cache_paths: dict[str, str]) -> bool:
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    hub = Path(cache_paths["HF_HUB_CACHE"])
    return hub.is_dir() and any(
        path.is_file() and path.suffix.lower() in suffixes
        for path in hub.rglob("*")
    )


def _cleanup_huggingface_download_cache(cache_paths: dict[str, str]) -> None:
    """Remove the temporary repository snapshot after conversion to the one .bin file."""
    for key in ("HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE"):
        directory = Path(cache_paths[key])
        if directory.exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def ensure_stage2_pretrained_checkpoints(
    project_root: Path | str,
    *,
    hf_token: str | None = None,
    pfm_keys: Iterable[str] = _STAGE2_PFM_KEYS,
) -> dict[str, Any]:
    """Install UNI2-h once as one persistent ``.bin`` file.

    The first call downloads the gated repository, converts the loaded model to a plain
    PyTorch state-dict binary, and removes the repository snapshot. Every later Colab or
    server session checks this binary first and loads it directly without downloading.
    """
    from puma.runtime import configure_project_checkpoint_cache

    selected_keys = tuple(dict.fromkeys(str(key) for key in pfm_keys))
    unknown = sorted(set(selected_keys) - set(PFM_SPECS))
    if unknown:
        raise KeyError(f"Unknown Stage-2 PFM key(s): {unknown}")
    if selected_keys != ("uni2_h",):
        raise ValueError("This Stage-2 revision supports only pfm_keys=('uni2_h',).")

    root = Path(project_root).expanduser().resolve()
    cache_paths = configure_project_checkpoint_cache(root)
    checkpoint = uni2_checkpoint_path(root)
    if _is_valid_checkpoint_file(checkpoint):
        print(f"REUSE UNI2-h binary: {checkpoint}")
        return {
            "models": {"uni2_h": "reused"},
            "checkpoint_file": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
        }

    has_cached_snapshot = _has_existing_huggingface_weights(cache_paths)
    if not hf_token and not has_cached_snapshot:
        raise RuntimeError(
            "UNI2-h is not installed yet and HF_TOKEN is unavailable. Accept the "
            "MahmoodLab/UNI2-h repository terms, add HF_TOKEN to Colab Secrets, "
            "and run the install cell once. Later sessions will not require the token."
        )

    action = "CONVERT cached UNI2-h snapshot" if has_cached_snapshot else "DOWNLOAD UNI2-h once"
    print(f"{action} and create: {checkpoint}")
    model: nn.Module | None = None
    try:
        model = _download_uni2_hub_model(hf_token)
        _atomic_save_state_dict(model, checkpoint)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not _is_valid_checkpoint_file(checkpoint):
        raise RuntimeError(f"UNI2-h binary creation failed: {checkpoint}")
    _cleanup_huggingface_download_cache(cache_paths)
    return {
        "models": {"uni2_h": "downloaded_and_converted"},
        "checkpoint_file": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def _runtime_local_checkpoint(persistent: Path) -> Path:
    """Copy the persistent Drive binary to Colab's local SSD once per runtime.

    Repeated folds and experiments then load from the local filesystem. If local disk is
    unavailable or too small, loading safely falls back to the persistent project file.
    """
    global _RUNTIME_CHECKPOINT_PATH
    if _RUNTIME_CHECKPOINT_PATH is not None and _is_valid_checkpoint_file(
        _RUNTIME_CHECKPOINT_PATH
    ):
        return _RUNTIME_CHECKPOINT_PATH
    if os.environ.get("PUMA_DISABLE_LOCAL_UNI2_COPY", "0") == "1":
        _RUNTIME_CHECKPOINT_PATH = persistent
        return persistent

    content_root = Path("/content")
    if not content_root.is_dir() or str(persistent).startswith(str(content_root)):
        _RUNTIME_CHECKPOINT_PATH = persistent
        return persistent
    local = content_root / "puma_uni2_checkpoint" / persistent.name
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        source_stat = persistent.stat()
        if not (
            local.is_file()
            and local.stat().st_size == source_stat.st_size
            and local.stat().st_mtime_ns == source_stat.st_mtime_ns
        ):
            required = int(source_stat.st_size * 1.05)
            if shutil.disk_usage(local.parent).free < required:
                _RUNTIME_CHECKPOINT_PATH = persistent
                return persistent
            print(f"COPY UNI2-h binary to local runtime SSD: {local}")
            temporary = local.with_name(local.name + ".partial")
            temporary.unlink(missing_ok=True)
            shutil.copy2(persistent, temporary)
            os.replace(temporary, local)
        _RUNTIME_CHECKPOINT_PATH = local
    except OSError as exc:
        print(f"Local UNI2-h copy unavailable ({exc}); loading from {persistent}")
        _RUNTIME_CHECKPOINT_PATH = persistent
    return _RUNTIME_CHECKPOINT_PATH


def _load_state_dict_fast(checkpoint: Path) -> dict[str, torch.Tensor]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        return torch.load(
            checkpoint, weights_only=True, mmap=True, **load_kwargs
        )
    except TypeError:
        # Older supported PyTorch builds may not expose weights_only or mmap.
        return torch.load(checkpoint, **load_kwargs)
    except RuntimeError:
        # Some non-local filesystems cannot be memory-mapped.
        try:
            return torch.load(checkpoint, weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(checkpoint, **load_kwargs)


def _load_uni2_from_binary(checkpoint: Path) -> nn.Module:
    state_dict = _load_state_dict_fast(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"UNI2-h checkpoint must contain a state_dict, got {type(state_dict)!r}."
        )

    # Meta-device construction avoids allocating and initializing a second 681M-parameter
    # CPU copy before assigning the memory-mapped weights. Fall back for older PyTorch.
    try:
        with torch.device("meta"):
            model = _create_uni2_architecture()
        model.load_state_dict(state_dict, strict=True, assign=True)
    except (TypeError, RuntimeError, NotImplementedError):
        model = _create_uni2_architecture()
        model.load_state_dict(state_dict, strict=True)
    del state_dict
    return model


def build_pfm_encoder(
    pfm_key: str,
    use_lora: bool = False,
    hf_token: str | None = None,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_last_blocks: int = 8,
) -> tuple[nn.Module, int]:
    """Build UNI2-h from the persistent project-local binary."""
    if pfm_key != "uni2_h":
        raise KeyError(
            f"This Stage-2 revision supports UNI2-h only, got {pfm_key!r}."
        )
    root = _project_root_from_environment()
    summary = ensure_stage2_pretrained_checkpoints(
        root, hf_token=hf_token, pfm_keys=("uni2_h",)
    )
    checkpoint = _runtime_local_checkpoint(Path(summary["checkpoint_file"]))
    try:
        model = _load_uni2_from_binary(checkpoint)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the local UNI2-h binary {checkpoint}. "
            "Delete the incomplete file and rerun the UNI2-h checkpoint cell. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    for parameter in model.parameters():
        parameter.requires_grad = False
    if use_lora:
        injected = inject_lora_last_blocks(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            last_n_blocks=lora_last_blocks,
        )
        model._puma_lora_injected_modules = injected
        # LoRA gradient checkpointing is optional because it trades speed for VRAM.
        checkpointing_enabled = os.environ.get(
            "PUMA_LORA_GRAD_CHECKPOINTING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        set_checkpointing = getattr(model, "set_grad_checkpointing", None)
        if callable(set_checkpointing):
            set_checkpointing(checkpointing_enabled)
        model._puma_lora_gradient_checkpointing = checkpointing_enabled
    return model, int(PFM_SPECS[pfm_key]["embedding_dim"])

def _resolve_token_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if isinstance(features, dict):
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            return torch.cat(
                [
                    features["x_norm_clstoken"][:, None, :],
                    features["x_norm_patchtokens"],
                ],
                dim=1,
            )
        for key in ("last_hidden_state", "features", "x"):
            if key in features:
                return features[key]
        features = next(iter(features.values()))
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Unsupported UNI2-h feature type: {type(features)!r}")
    return features


def _split_prefix_and_patch_tokens(
    features: Any, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = _resolve_token_tensor(features)
    if tokens.ndim != 3:
        raise ValueError(
            "Center-aware UNI2-h pooling requires token features shaped [B,N,C], "
            f"got {tuple(tokens.shape)}."
        )
    prefix_count = int(getattr(model, "num_prefix_tokens", 1))
    prefix_count = max(1, min(prefix_count, tokens.shape[1] - 1))
    cls_token = tokens[:, 0]
    patch_tokens = tokens[:, prefix_count:]
    if patch_tokens.shape[1] == 0:
        raise RuntimeError("UNI2-h returned no patch tokens after prefix removal.")
    grid = int(round(math.sqrt(patch_tokens.shape[1])))
    if grid * grid != patch_tokens.shape[1]:
        raise RuntimeError(
            "UNI2-h patch-token count must form a square grid; "
            f"got {patch_tokens.shape[1]} tokens."
        )
    return cls_token, patch_tokens


def pooled_feature_multiplier(pooling_key: str) -> int:
    if pooling_key in {"cls", "center"}:
        return 1
    if pooling_key == "cls_center_ring":
        return 3
    raise KeyError(f"Unknown Stage-2 pooling_key={pooling_key!r}.")


def pool_pfm_features(
    features: Any,
    pooling_key: str,
    model: nn.Module,
    *,
    center_size: int = 4,
) -> torch.Tensor:
    """Pool UNI2-h tokens around the known central candidate location."""
    cls_token, patch_tokens = _split_prefix_and_patch_tokens(features, model)
    batch, token_count, channels = patch_tokens.shape
    grid = int(round(math.sqrt(token_count)))
    patch_grid = patch_tokens.reshape(batch, grid, grid, channels)
    width = max(1, min(int(center_size), grid))
    start = (grid - width) // 2
    stop = start + width
    center = patch_grid[:, start:stop, start:stop].mean(dim=(1, 2))

    if pooling_key == "cls":
        return cls_token
    if pooling_key == "center":
        return center
    if pooling_key == "cls_center_ring":
        center_mask = torch.zeros(
            (grid, grid), dtype=torch.bool, device=patch_grid.device
        )
        center_mask[start:stop, start:stop] = True
        ring = patch_grid[:, ~center_mask].mean(dim=1)
        return torch.cat([cls_token, center, ring], dim=-1)
    raise KeyError(f"Unknown Stage-2 pooling_key={pooling_key!r}.")


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if len(features) < 2:
            return features.sum() * 0.0
        features = F.normalize(features, dim=-1)
        logits = features @ features.T / self.temperature
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        positives = labels[:, None].eq(labels[None, :]) & ~eye
        log_denominator = torch.logsumexp(
            logits.masked_fill(eye, float("-inf")), dim=1, keepdim=True
        )
        log_probability = logits - log_denominator
        positive_count = positives.sum(1)
        valid = positive_count > 0
        per_anchor = -(
            log_probability.masked_fill(~positives, 0.0).sum(1)
            / positive_count.clamp_min(1).float()
        )
        return per_anchor[valid].mean() if bool(valid.any()) else per_anchor.sum() * 0.0


class ScaleFusePFM(nn.Module):
    """Fixed-multiscale UNI2-h classifier for A1_IFCRN_PP candidates."""

    VIEW_FOV: dict[str, int] = {"V2": 64, "V3": 128, "V4": 256}

    def __init__(
        self,
        cfg: Stage2ModelConfig,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate_config()
        self.encoder, base_dim = build_pfm_encoder(
            cfg.pfm_key,
            cfg.use_lora,
            hf_token,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_last_blocks=cfg.lora_last_blocks,
        )
        self.encoder_trainable = any(
            parameter.requires_grad for parameter in self.encoder.parameters()
        )
        pooled_dim = base_dim * pooled_feature_multiplier(cfg.pooling_key)
        hidden = cfg.hidden_dim
        self.hidden_dim = hidden
        self.view_projections = nn.ModuleDict(
            {
                view: nn.Sequential(
                    nn.LayerNorm(pooled_dim),
                    nn.Linear(pooled_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for view in cfg.views
            }
        )
        self.view_embeddings = nn.ParameterDict(
            {
                view: nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
                for view in cfg.views
            }
        )
        self.fov_projection = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.geometry_mlp = (
            nn.Sequential(
                nn.LayerNorm(STAGE2_GEOMETRY_DIM),
                nn.Linear(STAGE2_GEOMETRY_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            if cfg.use_geometry
            else None
        )
        heads = min(8, max(1, hidden // 64))
        while hidden % heads:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=cfg.fusion_layers, enable_nested_tensor=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.validity_classifier = nn.Linear(hidden, 1)
        self.type_classifier = nn.Linear(hidden, len(PUMA_CLASS_NAMES))
        self.supcon_projection = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 128)
        )
        self.supcon = SupConLoss(temperature=0.1)

        if cfg.loss_key == "TYPE_BALANCED":
            for module in (self.validity_classifier, self.supcon_projection):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        elif cfg.loss_key == "HIERARCHICAL":
            for parameter in self.supcon_projection.parameters():
                parameter.requires_grad = False

    def _validate_config(self) -> None:
        if self.cfg.pfm_key != "uni2_h":
            raise ValueError("ScaleFusePFM supports UNI2-h only.")
        unknown = sorted(set(self.cfg.views) - set(self.VIEW_FOV))
        if unknown or not self.cfg.views:
            raise ValueError(
                f"Stage-2 views must be a non-empty subset of V2/V3/V4, got {self.cfg.views}."
            )
        if self.cfg.interface_key != "Fixed-MV":
            raise ValueError("A1_IFCRN_PP requires interface_key='Fixed-MV'.")
        if self.cfg.loss_key not in {
            "TYPE_BALANCED", "HIERARCHICAL", "HIERARCHICAL_SUPCON"
        }:
            raise ValueError(f"Unsupported loss_key={self.cfg.loss_key!r}.")
        if self.cfg.type_loss_key not in {"BALANCED_SOFTMAX", "CE", "CB_CE", "CB_FOCAL"}:
            raise ValueError(
                f"Unsupported type_loss_key={self.cfg.type_loss_key!r}."
            )
        if self.cfg.validity_loss_key != "BCE":
            raise ValueError(
                "V13.2 implements BCE validity loss only; "
                f"got validity_loss_key={self.cfg.validity_loss_key!r}."
            )
        pooled_feature_multiplier(self.cfg.pooling_key)
        if self.cfg.encoder_micro_batch_size <= 0:
            raise ValueError("encoder_micro_batch_size must be positive.")
        if self.cfg.type_loss_weight <= 0 or self.cfg.validity_loss_weight < 0:
            raise ValueError("Stage-2 loss weights must be positive/non-negative.")
        if not 0.0 < self.cfg.sampler_positive_fraction <= 1.0:
            raise ValueError("sampler_positive_fraction must be in (0,1].")
        if not 0.0 <= self.cfg.sampler_balanced_positive_fraction <= 1.0:
            raise ValueError("sampler_balanced_positive_fraction must be in [0,1].")
        if self.cfg.sampler_max_repeats < 1 or self.cfg.sampler_tail_max_repeats < 1:
            raise ValueError("sampler repeat caps must be >= 1.")
        if self.cfg.hard_negative_start_phase_epoch < 1 or self.cfg.checkpoint_selection_start_phase_epoch < 1:
            raise ValueError("V13 phase-epoch controls must be >= 1.")

    def train(self, mode: bool = True) -> "ScaleFusePFM":
        super().train(mode)
        if not self.encoder_trainable:
            self.encoder.eval()
        return self

    def _forward_encoder(self, image: torch.Tensor) -> Any:
        forward_features = getattr(self.encoder, "forward_features", None)
        return forward_features(image) if callable(forward_features) else self.encoder(image)

    def extract_view_features(self, image: torch.Tensor) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        micro = min(self.cfg.encoder_micro_batch_size, max(len(image), 1))
        context = torch.enable_grad if self.encoder_trainable else torch.no_grad
        with context():
            for start in range(0, len(image), micro):
                features = self._forward_encoder(image[start : start + micro])
                chunks.append(
                    pool_pfm_features(
                        features, self.cfg.pooling_key, self.encoder
                    )
                )
        if not chunks:
            multiplier = pooled_feature_multiplier(self.cfg.pooling_key)
            base_dim = int(PFM_SPECS[self.cfg.pfm_key]["embedding_dim"])
            return image.new_empty((0, base_dim * multiplier))
        return torch.cat(chunks, dim=0)

    def project_view_features(
        self, features: torch.Tensor, view_key: str
    ) -> torch.Tensor:
        if view_key not in self.view_projections:
            raise KeyError(f"View {view_key!r} is not enabled in {self.cfg.views}.")
        return self.view_projections[view_key](features)

    def encode_view(self, image: torch.Tensor, view_key: str) -> torch.Tensor:
        return self.project_view_features(
            self.extract_view_features(image), view_key
        )

    def fuse_projected_views(
        self,
        projected_views: dict[str, torch.Tensor],
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for view in self.cfg.views:
            if view not in projected_views:
                raise KeyError(f"Missing projected Stage-2 view {view!r}.")
            batch = projected_views[view].shape[0]
            log_fov = math.log(float(self.VIEW_FOV[view]) / 128.0)
            fov = geometry.new_full((batch, 1), log_fov)
            scale_token = self.fov_projection(fov)[:, None, :]
            tokens.append(
                projected_views[view][:, None, :]
                + self.view_embeddings[view]
                + scale_token
            )
        if self.geometry_mlp is not None:
            tokens.append(self.geometry_mlp(geometry)[:, None, :])
        if not tokens:
            raise RuntimeError("ScaleFusePFM requires at least one view token.")
        batch = tokens[0].shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        return self.fusion(torch.cat([cls, *tokens], dim=1))[:, 0]

    def fuse_candidates(
        self,
        views: dict[str, torch.Tensor],
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        projected = {
            view: self.encode_view(views[view], view)
            for view in self.cfg.views
        }
        return self.fuse_projected_views(projected, geometry)

    def classify_fused(
        self,
        fused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.norm(fused)
        outputs: dict[str, torch.Tensor] = {
            "type_logits": self.type_classifier(fused)
        }
        if self.cfg.loss_key != "TYPE_BALANCED":
            outputs["validity_logits"] = self.validity_classifier(fused).squeeze(-1)
        if self.cfg.loss_key == "HIERARCHICAL_SUPCON":
            outputs["contrastive_embedding"] = F.normalize(
                self.supcon_projection(fused), dim=-1
            )
        return outputs

    def forward(
        self,
        views: dict[str, torch.Tensor],
        geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.fuse_candidates(views, geometry)
        return self.classify_fused(fused)


def hierarchical_probabilities(
    outputs: dict[str, torch.Tensor], loss_key: str
) -> torch.Tensor:
    """Return 10 type probabilities plus a REJECT probability."""
    type_probability = outputs["type_logits"].softmax(-1)
    if loss_key == "TYPE_BALANCED":
        reject = torch.zeros(
            (len(type_probability), 1),
            dtype=type_probability.dtype,
            device=type_probability.device,
        )
        return torch.cat([type_probability, reject], dim=-1)
    validity = outputs["validity_logits"].sigmoid()
    return torch.cat(
        [validity[:, None] * type_probability, (1.0 - validity)[:, None]],
        dim=-1,
    )


def stage2_probability_components(
    probabilities: torch.Tensor,
    loss_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predicted type, type confidence, validity, and joint confidence."""
    if probabilities.ndim != 2 or probabilities.shape[1] != REJECT_CLASS_ID + 1:
        raise ValueError(
            "Stage-2 probabilities must have shape [N, 11], got "
            f"{tuple(probabilities.shape)}."
        )
    type_mass = probabilities[:, :REJECT_CLASS_ID]
    if loss_key == "TYPE_BALANCED":
        validity = torch.ones(
            len(probabilities), dtype=probabilities.dtype, device=probabilities.device
        )
        normalized_type = type_mass
    else:
        validity = (1.0 - probabilities[:, REJECT_CLASS_ID]).clamp(0.0, 1.0)
        normalized_type = type_mass / validity[:, None].clamp_min(1e-8)
    type_confidence, predicted_type = normalized_type.max(-1)
    joint_confidence = type_mass.gather(1, predicted_type[:, None]).squeeze(1)
    return predicted_type, type_confidence, validity, joint_confidence


def decode_stage2_probabilities(
    probabilities: torch.Tensor,
    loss_key: str,
    validity_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode type and validity independently instead of using 11-way argmax."""
    predicted_type, _, validity, joint_confidence = stage2_probability_components(
        probabilities, loss_key
    )
    if loss_key == "TYPE_BALANCED":
        return predicted_type, joint_confidence
    is_valid = validity >= float(validity_threshold)
    predicted = torch.where(
        is_valid,
        predicted_type,
        torch.full_like(predicted_type, REJECT_CLASS_ID),
    )
    confidence = torch.where(is_valid, joint_confidence, 1.0 - validity)
    return predicted, confidence


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    if not 0.0 <= positive_alpha <= 1.0:
        raise ValueError("positive_alpha must be in [0,1].")
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = (
        positive_alpha * targets
        + (1.0 - positive_alpha) * (1.0 - targets)
    )
    return (alpha_t * (1.0 - p_t).pow(gamma) * cross_entropy).mean()


def effective_number_weights(
    class_counts: torch.Tensor, beta: float = 0.999
) -> torch.Tensor:
    counts = class_counts.float().clamp_min(1.0)
    weights = (1.0 - beta) / (
        1.0 - torch.pow(torch.full_like(counts, beta), counts)
    )
    return weights / weights.mean().clamp_min(1e-8)






def split_optimizer_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return head parameters and LoRA parameters without duplication."""
    head: list[nn.Parameter] = []
    lora: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora.append(parameter)
        else:
            head.append(parameter)
    if not head:
        raise RuntimeError("Stage-2 model has no trainable classifier/fusion parameters.")
    return head, lora


def build_stage2_model(
    cfg: Stage2ModelConfig, hf_token: str | None = None
) -> ScaleFusePFM:
    return ScaleFusePFM(cfg, hf_token=hf_token)
