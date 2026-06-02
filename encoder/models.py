from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUCLEI_CLASSES, TISSUE_CLASSES, TrainConfig

_CHANNEL_LIKE = {
    32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 640,
    768, 960, 1024, 1280, 1536, 2048, 2560, 4096,
}


class LoRALinear(nn.Module):
    """
    Drop-in LoRA adapter for nn.Linear.

    The original linear layer is kept frozen. Only lora_A and lora_B are
    trainable. This keeps E6/E7 as encoder LoRA experiments without changing
    the decoder/probe architecture.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'LoRA rank must be positive, got {rank}')

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _set_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_vit_encoder(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> int:
    """
    Apply LoRA to transformer Linear layers.

    This targets Linear layers under transformer blocks, which covers the
    attention and MLP projections used by UNI2-H and Virchow2 in timm. It avoids
    touching the external probe decoder because this function is called only on
    FrozenTimmBackbone.model.
    """
    replaced = 0
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        in_transformer_block = 'blocks.' in lname or '.blocks.' in lname
        looks_like_vit_projection = any(
            token in lname
            for token in (
                'attn.qkv', 'attn.proj', 'qkv', 'proj', 'fc1', 'fc2',
                'w12', 'w3', 'mlp', 'linear1', 'linear2',
            )
        )
        if in_transformer_block and looks_like_vit_projection:
            candidates.append((name, module))

    for name, module in candidates:
        _set_module_by_name(
            model,
            name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            'LoRA was requested, but no transformer Linear layers were found. '
            'Check the timm model structure or update apply_lora_to_vit_encoder targets.'
        )
    return replaced


def _to_nchw(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 4:
        # CNN features are usually NCHW. Some ViT features can be NHWC.
        if feature.shape[-1] in _CHANNEL_LIKE and feature.shape[1] not in _CHANNEL_LIKE:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature.contiguous()

    if feature.ndim == 3:
        # Token output: [B, tokens, C]. Remove cls/register tokens if needed
        # until the remaining tokens form a square grid.
        b, n, c = feature.shape
        side = int(math.sqrt(n))

        if side * side != n:
            found = False
            for skip in range(1, min(32, n)):
                n2 = n - skip
                side = int(math.sqrt(n2))
                if side * side == n2:
                    feature = feature[:, skip:, :]
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f'Cannot reshape token output {tuple(feature.shape)} into a square feature map'
                )

        b, n, c = feature.shape
        side = int(math.sqrt(n))
        return feature.transpose(1, 2).reshape(b, c, side, side).contiguous()

    if feature.ndim == 2:
        # Global embedding. This is still usable by the probe, but it gives one
        # feature location per tile.
        return feature[:, :, None, None].contiguous()

    raise RuntimeError(f'Expected 2D/3D/4D feature output, got shape {tuple(feature.shape)}')


def _extract_from_output(y: Any) -> torch.Tensor:
    if isinstance(y, dict):
        for key in (
            'x_norm_patchtokens',
            'patch_tokens',
            'last_hidden_state',
            'features',
            'tokens',
        ):
            if key in y and torch.is_tensor(y[key]):
                return _to_nchw(y[key])

        tensors = [v for v in y.values() if torch.is_tensor(v)]
        if not tensors:
            raise RuntimeError('Model returned a dict without tensor values')
        return _to_nchw(tensors[-1])

    if isinstance(y, (list, tuple)):
        if not y:
            raise RuntimeError('Model returned an empty list/tuple')
        return _to_nchw(y[-1])

    if torch.is_tensor(y):
        return _to_nchw(y)

    raise RuntimeError(f'Unsupported model output type: {type(y)}')


def _extract_model_output(
    model: nn.Module,
    x: torch.Tensor,
    prefer_forward_features: bool = False,
) -> torch.Tensor:
    if prefer_forward_features and hasattr(model, 'forward_features'):
        return _extract_from_output(model.forward_features(x))
    return _extract_from_output(model(x))


def _is_foundation_encoder(model_name: str) -> bool:
    name = model_name.lower()
    return (
        'uni' in name
        or 'virchow' in name
        or 'mahmoodlab' in name
        or 'paige-ai' in name
    )


def _stitch_tile_features(
    tile_feats: torch.Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    # tile_feats: [B * grid_h * grid_w, C, fh, fw]
    n, c, fh, fw = tile_feats.shape
    expected = batch_size * grid_h * grid_w

    if n != expected:
        raise RuntimeError(f'Tile feature count mismatch: expected {expected}, got {n}')

    x = tile_feats.view(batch_size, grid_h, grid_w, c, fh, fw)
    # B, gh, gw, C, fh, fw -> B, C, gh, fh, gw, fw -> B, C, gh*fh, gw*fw
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(batch_size, c, grid_h * fh, grid_w * fw)


def _create_foundation_model(model_name: str, pretrained: bool) -> nn.Module:
    """
    Create pathology foundation encoders with the correct architecture kwargs.

    UNI2-H cannot be created safely with only:
        timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True)
    because timm then uses wrong ViT assumptions and can fail during checkpoint
    positional/token reshaping.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError('Please install timm: pip install timm') from exc

    name = model_name.lower().replace('_', '-')

    if 'mahmoodlab/uni2-h' in name:
        return timm.create_model(
            'hf-hub:MahmoodLab/UNI2-h',
            pretrained=pretrained,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )

    if 'paige-ai/virchow2' in name:
        return timm.create_model(
            'hf-hub:paige-ai/Virchow2',
            pretrained=pretrained,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
    )


class FrozenTimmBackbone(nn.Module):
    """
    Frozen timm backbone wrapper.

    CNN encoders receive the full ROI. Foundation pathology ViTs receive
    256x256 tiles. Each tile is resized to 224x224 before UNI2-H / Virchow2,
    then tile features are stitched back into one spatial feature map.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        frozen: bool = True,
        tile_size: int = 256,
        tile_batch: int = 8,
        model_input_size: int = 224,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError('Please install timm: pip install timm') from exc

        self.model_name = model_name
        self.frozen = frozen
        self.tile_size = int(tile_size)
        self.tile_batch = int(tile_batch)
        self.model_input_size = int(model_input_size)
        self.is_foundation = _is_foundation_encoder(model_name)
        self.use_lora = bool(use_lora)
        self.uses_features_only = False

        if self.is_foundation:
            self.model = _create_foundation_model(model_name, pretrained=pretrained)
            self.uses_features_only = False
        else:
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(-1,),
                )
                self.uses_features_only = True
            except Exception:
                self.model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                )
                self.uses_features_only = False

        if frozen or self.use_lora:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.lora_layers = 0
        if self.use_lora:
            if not self.is_foundation:
                raise ValueError('LoRA is intended for foundation ViT encoders only in E6/E7.')
            self.lora_layers = apply_lora_to_vit_encoder(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            # LoRA needs gradients through the frozen backbone graph, so this
            # wrapper must not use torch.no_grad() when use_lora=True.
            self.frozen = False
        elif frozen:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def _forward_model(self, x: torch.Tensor) -> torch.Tensor:
        prefer_forward_features = self.is_foundation or not self.uses_features_only
        if self.frozen:
            with torch.no_grad():
                return _extract_model_output(
                    self.model,
                    x,
                    prefer_forward_features=prefer_forward_features,
                )
        return _extract_model_output(
            self.model,
            x,
            prefer_forward_features=prefer_forward_features,
        )

    def _forward_tiled(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ts = self.tile_size

        if h % ts != 0 or w % ts != 0:
            raise ValueError(
                f'Image size {h}x{w} must be divisible by foundation_tile_size={ts}'
            )

        gh, gw = h // ts, w // ts

        tiles = (
            x.unfold(2, ts, ts)
            .unfold(3, ts, ts)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(b * gh * gw, c, ts, ts)
        )

        chunks = []
        for start in range(0, tiles.shape[0], self.tile_batch):
            chunk = tiles[start:start + self.tile_batch]

            if self.model_input_size > 0 and chunk.shape[-1] != self.model_input_size:
                chunk = F.interpolate(
                    chunk,
                    size=(self.model_input_size, self.model_input_size),
                    mode='bilinear',
                    align_corners=False,
                )

            chunks.append(self._forward_model(chunk))

        tile_feats = torch.cat(chunks, dim=0)
        return _stitch_tile_features(
            tile_feats,
            batch_size=b,
            grid_h=gh,
            grid_w=gw,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_foundation:
            return self._forward_tiled(x)
        return self._forward_model(x)


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        y = self.net(x)
        return F.interpolate(y, size=out_size, mode='bilinear', align_corners=False)


class PumaEncoderProbe(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.primary = FrozenTimmBackbone(
            cfg.encoder_name,
            pretrained=cfg.pretrained,
            frozen=cfg.freeze_encoders,
            tile_size=cfg.foundation_tile_size,
            tile_batch=cfg.foundation_tile_batch,
            model_input_size=cfg.foundation_model_input_size,
            use_lora=cfg.use_lora,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
        )

        self.auxiliary = None
        if cfg.encoder_kind == 'fusion':
            if cfg.aux_encoder_name is None:
                raise ValueError('Fusion experiment requires cfg.aux_encoder_name')
            self.auxiliary = FrozenTimmBackbone(
                cfg.aux_encoder_name,
                pretrained=cfg.pretrained,
                frozen=cfg.freeze_encoders,
                tile_size=cfg.foundation_tile_size,
                tile_batch=cfg.foundation_tile_batch,
                model_input_size=cfg.foundation_model_input_size,
                use_lora=False,
            )

        self.primary_proj = nn.Sequential(
            nn.LazyConv2d(cfg.decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.decoder_channels),
            nn.ReLU(inplace=True),
        )

        if self.auxiliary is not None:
            self.aux_proj = nn.Sequential(
                nn.LazyConv2d(cfg.decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(cfg.decoder_channels),
                nn.ReLU(inplace=True),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(
                    cfg.decoder_channels * 2,
                    cfg.decoder_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cfg.decoder_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.aux_proj = None
            self.fuse = nn.Identity()

        self.tissue_head = ConvHead(cfg.decoder_channels, cfg.head_channels, len(TISSUE_CLASSES))
        self.nuclei_fg_head = ConvHead(cfg.decoder_channels, cfg.head_channels, 2)
        self.nuclei_class_head = ConvHead(cfg.decoder_channels, cfg.head_channels, len(NUCLEI_CLASSES))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out_size = x.shape[-2:]
        p = self.primary_proj(self.primary(x))

        if self.auxiliary is not None:
            a = self.aux_proj(self.auxiliary(x))
            a = F.interpolate(a, size=p.shape[-2:], mode='bilinear', align_corners=False)
            feat = self.fuse(torch.cat([p, a], dim=1))
        else:
            feat = self.fuse(p)

        return {
            'tissue': self.tissue_head(feat, out_size),
            'nuclei_fg': self.nuclei_fg_head(feat, out_size),
            'nuclei_class': self.nuclei_class_head(feat, out_size),
        }
