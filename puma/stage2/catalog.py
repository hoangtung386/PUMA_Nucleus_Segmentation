from __future__ import annotations

from puma.config import Stage2ModelConfig

VERSION13_EXPERIMENTS: tuple[str, ...] = (
    "V13_01_META_NEW_SPLIT_FROZEN",
    "V13_02_META_CONTEXT_NEW_SPLIT_FROZEN",
    "V13_03_META_CONTEXT_CBFOCAL_FROZEN",
    "V13_04_META_CONTEXT_CBCE_FROZEN",
    "V13_05_META_CONTEXT_RAREBOOST_FROZEN",
    "V13_06_META_CONTEXT_LORA_R8_B4",
)

VERSION13_EXPERIMENT_PURPOSE: dict[str, str] = {
    VERSION13_EXPERIMENTS[0]: (
        "META baseline: V64 + V128 with geometry on the fixed development split."
    ),
    VERSION13_EXPERIMENTS[1]: (
        "Architecture consolidation: combine META geometry with V256 tissue context."
    ),
    VERSION13_EXPERIMENTS[2]: (
        "Rare-class loss test: class-balanced focal type loss on META+CONTEXT."
    ),
    VERSION13_EXPERIMENTS[3]: (
        "Rare-class loss test: effective-number class-balanced cross entropy on META+CONTEXT."
    ),
    VERSION13_EXPERIMENTS[4]: (
        "Sampling test: stronger class-balanced positive sampling and larger repeat budget for tail classes."
    ),
    VERSION13_EXPERIMENTS[5]: (
        "Representation adaptation: LoRA rank 8 in the last 4 UNI2-h blocks on META+CONTEXT."
    ),
}


def _v13(
    name: str,
    *,
    views: tuple[str, ...] = ("V2", "V3", "V4"),
    use_geometry: bool = True,
    type_loss_key: str = "BALANCED_SOFTMAX",
    validity_loss_key: str = "BCE",
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_last_blocks: int = 4,
    sampler_positive_fraction: float = 2.0 / 3.0,
    sampler_balanced_positive_fraction: float = 0.50,
    sampler_max_repeats: int = 4,
    sampler_tail_max_repeats: int = 4,
    type_loss_weight: float = 1.0,
    validity_loss_weight: float = 1.0,
) -> Stage2ModelConfig:
    return Stage2ModelConfig(
        name=name,
        views=views,
        pooling_key="cls_center_ring",
        schedule_key="GT_POS+OOF_POS+OOF_ALL",
        loss_key="HIERARCHICAL",
        use_geometry=use_geometry,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_last_blocks=lora_last_blocks,
        selection_metric="macro_f1",
        # Shared V13 encoder batch target.
        encoder_micro_batch_size=256,
        type_loss_key=type_loss_key,
        validity_loss_key=validity_loss_key,
        type_loss_weight=type_loss_weight,
        validity_loss_weight=validity_loss_weight,
        sampler_positive_fraction=sampler_positive_fraction,
        sampler_balanced_positive_fraction=sampler_balanced_positive_fraction,
        sampler_max_repeats=sampler_max_repeats,
        sampler_tail_max_repeats=sampler_tail_max_repeats,
        hard_negative_start_phase_epoch=6,
        checkpoint_selection_start_phase_epoch=6,
        use_stain_augmentation=True,
    )


def stage2_experiment_registry() -> dict[str, Stage2ModelConfig]:
    """Return the six Stage-2 V13 experiments on one fixed split."""
    configs = (
        _v13(VERSION13_EXPERIMENTS[0], views=("V2", "V3"), use_geometry=True),
        _v13(VERSION13_EXPERIMENTS[1]),
        _v13(VERSION13_EXPERIMENTS[2], type_loss_key="CB_FOCAL"),
        _v13(VERSION13_EXPERIMENTS[3], type_loss_key="CB_CE"),
        _v13(
            VERSION13_EXPERIMENTS[4],
            sampler_positive_fraction=0.75,
            sampler_balanced_positive_fraction=0.75,
            sampler_max_repeats=6,
            sampler_tail_max_repeats=10,
        ),
        _v13(
            VERSION13_EXPERIMENTS[5],
            use_lora=True,
            lora_rank=8,
            lora_last_blocks=4,
        ),
    )
    if tuple(cfg.name for cfg in configs) != VERSION13_EXPERIMENTS:
        raise RuntimeError("Version-13 Stage-2 experiment catalog is out of order.")
    return {cfg.name: cfg for cfg in configs}


def stage2_experiment_groups() -> dict[str, tuple[str, ...]]:
    registry = stage2_experiment_registry()
    return {
        "frozen": tuple(name for name, cfg in registry.items() if not cfg.use_lora),
        "lora": tuple(name for name, cfg in registry.items() if cfg.use_lora),
        "all": tuple(registry),
    }
