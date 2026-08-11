from __future__ import annotations

from dataclasses import replace

from puma.config import Stage2ModelConfig

VERSION132_EXPERIMENTS: tuple[str, ...] = (
    "V13_2_01_META_CONTROL_BS",
    "V13_2_02_META_RARE_BS",
    "V13_2_03_META_RARE_CE",
    "V13_2_04_META_CONTEXT_RARE_BS",
)

VERSION132_EXPERIMENT_PURPOSE: dict[str, str] = {
    VERSION132_EXPERIMENTS[0]: (
        "Control: V64+V128 META geometry with Balanced Softmax and the moderate V13.1-style sampler."
    ),
    VERSION132_EXPERIMENTS[1]: (
        "Primary V13.2 model: V64+V128 META geometry, Balanced Softmax, strong case-aware rare exposure, "
        "hard rare positives and hard rejects."
    ),
    VERSION132_EXPERIMENTS[2]: (
        "Loss sanity check: identical strong rare-exposure policy to experiment 02 but plain CE type loss."
    ),
    VERSION132_EXPERIMENTS[3]: (
        "Context ablation: identical to experiment 02 with strong rare exposure and Balanced Softmax, "
        "but adds the V256 context view (V64+V128+V256)."
    ),
}


def _base(name: str) -> Stage2ModelConfig:
    return Stage2ModelConfig(
        name=name,
        views=("V2", "V3"),
        use_geometry=True,
        use_lora=False,
        loss_key="HIERARCHICAL",
        schedule_key="GT_POS+OOF_POS+OOF_ALL",
        interface_key="Fixed-MV",
        type_loss_key="BALANCED_SOFTMAX",
        validity_loss_key="BCE",
        use_stain_augmentation=True,
        encoder_micro_batch_size=256,
    )


def stage2_experiment_registry() -> dict[str, Stage2ModelConfig]:
    control = replace(
        _base(VERSION132_EXPERIMENTS[0]),
        use_strong_rare_sampling=False,
        sampler_balanced_positive_fraction=0.50,
        sampler_max_repeats=4,
        sampler_tail_max_repeats=4,
        rare_quota_gt_per_class=0,
        rare_quota_oof_pos_per_class=0,
        rare_quota_oof_all_per_class=0,
        hard_rare_fraction=0.0,
    )
    main = _base(VERSION132_EXPERIMENTS[1])
    ce = replace(_base(VERSION132_EXPERIMENTS[2]), type_loss_key="CE")
    context = replace(_base(VERSION132_EXPERIMENTS[3]), views=("V2", "V3", "V4"))
    configs = (control, main, ce, context)
    if tuple(cfg.name for cfg in configs) != VERSION132_EXPERIMENTS:
        raise RuntimeError("V13.2 Stage-2 experiment catalog is out of order.")
    return {cfg.name: cfg for cfg in configs}


def stage2_experiment_groups() -> dict[str, tuple[str, ...]]:
    registry = stage2_experiment_registry()
    return {
        "screening": tuple(registry),
        "recommended": (VERSION132_EXPERIMENTS[1],),
        "all": tuple(registry),
    }
