from __future__ import annotations

from puma.config import RuntimeConfig
from puma.stage2.catalog import VERSION13_EXPERIMENTS, stage2_experiment_registry

VERSION13_PARALLEL_RUNS = 1
VERSION13_LORA_PARALLEL_RUNS = 1
VERSION13_EFFECTIVE_BATCH_SIZE = 256
VERSION13_STAGE2_MICRO_BATCH_SIZE = 256
VERSION13_ENCODER_MICRO_BATCH_SIZE = 256


def validate_version13_plan(
    runtime: RuntimeConfig,
    experiment_names: list[str] | tuple[str, ...],
    *,
    parallel_runs: int = VERSION13_PARALLEL_RUNS,
    lora_parallel_runs: int = VERSION13_LORA_PARALLEL_RUNS,
    require_requested_batch_sizes: bool = True,
) -> dict:
    registry = stage2_experiment_registry()
    unknown = sorted(set(experiment_names) - set(registry))
    if unknown:
        raise KeyError(f"Unknown Version-13 experiment(s): {unknown}")
    if not experiment_names:
        raise ValueError("Select at least one Version-13 Stage-2 experiment.")
    if len(experiment_names) > len(VERSION13_EXPERIMENTS):
        raise ValueError("Version 13 defines only six controlled experiments.")
    if parallel_runs < 1 or lora_parallel_runs < 1:
        raise ValueError("Parallel-run counts must be >= 1.")
    if require_requested_batch_sizes:
        if runtime.training.effective_batch_size != VERSION13_EFFECTIVE_BATCH_SIZE:
            raise ValueError(
                f"V13 requested effective_batch_size={VERSION13_EFFECTIVE_BATCH_SIZE}; "
                f"got {runtime.training.effective_batch_size}. Set require_requested_batch_sizes=False to override."
            )
        if runtime.training.stage2_micro_batch_size != VERSION13_STAGE2_MICRO_BATCH_SIZE:
            raise ValueError(
                f"V13 requested stage2_micro_batch_size={VERSION13_STAGE2_MICRO_BATCH_SIZE}; "
                f"got {runtime.training.stage2_micro_batch_size}."
            )
        bad = {
            name: registry[name].encoder_micro_batch_size
            for name in experiment_names
            if registry[name].encoder_micro_batch_size != VERSION13_ENCODER_MICRO_BATCH_SIZE
        }
        if bad:
            raise ValueError(f"V13 encoder_micro_batch_size must be 256 for requested runs: {bad}")
    # Imported here to avoid a package-level circular dependency.
    from puma.training.stage2_v13 import ensure_v13_split
    split = ensure_v13_split(runtime, force=False, check_sources=False)
    return {
        "train_roi_count": len(split["train_roi_indices"]),
        "val_roi_count": len(split["val_roi_indices"]),
        "split_hash": split["split_hash"],
        "parallel_runs": int(parallel_runs),
        "lora_parallel_runs": int(lora_parallel_runs),
        "effective_batch_size": runtime.training.effective_batch_size,
        "stage2_micro_batch_size": runtime.training.stage2_micro_batch_size,
        "encoder_micro_batch_size": VERSION13_ENCODER_MICRO_BATCH_SIZE,
    }
