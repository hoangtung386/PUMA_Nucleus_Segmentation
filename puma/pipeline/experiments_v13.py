from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from puma.config import RuntimeConfig
from puma.models.stage2 import ensure_stage2_pretrained_checkpoints
from puma.pipeline.oof import validate_full_oof
from puma.stage2.catalog import VERSION13_EXPERIMENTS, stage2_experiment_groups, stage2_experiment_registry
from puma.stage2.runner_v13 import run_v13_jobs
from puma.training.stage2_v13 import V13_SPLIT_NAME, ensure_v13_split
from puma.utils import atomic_write_json, resolve_artifact_reference, utc_now_iso


def _resolve_names(
    experiments_to_run: Iterable[str] | None,
    experiment_group: str | None,
) -> tuple[str, ...]:
    registry = stage2_experiment_registry()
    if experiments_to_run is not None and experiment_group is not None:
        raise ValueError("Provide either experiments_to_run or experiment_group, not both.")
    if experiments_to_run is not None:
        names = tuple(str(name) for name in experiments_to_run)
    else:
        group = experiment_group or "all"
        groups = stage2_experiment_groups()
        if group not in groups:
            raise KeyError(f"Unknown V13 experiment group {group!r}: {sorted(groups)}")
        names = groups[group]
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise KeyError(f"Unknown V13 Stage-2 experiment(s): {unknown}")
    if not names:
        raise ValueError("At least one V13 Stage-2 experiment must be selected.")
    if len(names) > len(VERSION13_EXPERIMENTS):
        raise ValueError("V13 defines only six controlled experiments.")
    return names


def aggregate_v13_results(
    runtime: RuntimeConfig,
    experiments: Iterable[str] | None = None,
) -> pd.DataFrame:
    path = runtime.paths.stage2_file("stage2_v13_results.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    keys = [c for c in ("experiment", "split", "seed") if c in frame.columns]
    frame = frame.drop_duplicates(subset=keys, keep="last")
    frame = frame[frame["status"].astype(str) == "completed"].copy()
    frame = frame[frame["split"].astype(str) == V13_SPLIT_NAME]
    if experiments is not None:
        frame = frame[frame["experiment"].astype(str).isin(tuple(experiments))]
    if frame.empty:
        return frame
    seeds = {int(seed) for seed in runtime.training.seeds}
    complete: list[str] = []
    for name, rows in frame.groupby("experiment"):
        observed = set(pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int).tolist())
        if seeds.issubset(observed):
            complete.append(str(name))
    frame = frame[frame["experiment"].astype(str).isin(complete)]
    if frame.empty:
        return frame

    metric_columns = [
        c for c in frame.columns
        if c in {
            "macro_f1", "sum_f1", "conditional_type_macro_f1",
            "conditional_type_macro_f1_present", "conditional_type_accuracy",
            "candidate_accuracy", "reject_precision", "reject_recall", "ece",
            "validity_threshold", "val_best_metric", "duration_minutes", "peak_vram_mb",
            "parameters_total", "parameters_trainable", "best_epoch",
        }
        or c.startswith("f1_nuclei_")
        or c.startswith("conditional_type_f1_class_")
        or c.startswith("conditional_type_support_class_")
    ]
    for c in metric_columns:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    aggregate = frame.groupby("experiment", as_index=False)[metric_columns].mean(numeric_only=True)
    if {"reject_precision", "reject_recall"}.issubset(aggregate.columns):
        p = aggregate["reject_precision"].fillna(0.0)
        r = aggregate["reject_recall"].fillna(0.0)
        aggregate["reject_f1"] = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return aggregate


def v13_experiment_status(runtime: RuntimeConfig) -> pd.DataFrame:
    registry = stage2_experiment_registry()
    path = runtime.paths.stage2_file("stage2_v13_results.csv")
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    seeds = {int(seed) for seed in runtime.training.seeds}
    for name in registry:
        subset = frame[frame.get("experiment", pd.Series(dtype=str)).astype(str) == name].copy() if not frame.empty else pd.DataFrame()
        if not subset.empty:
            keys = [c for c in ("experiment", "split", "seed") if c in subset]
            subset = subset.drop_duplicates(subset=keys, keep="last")
            subset = subset[
                (subset["status"].astype(str) == "completed")
                & (subset["split"].astype(str) == V13_SPLIT_NAME)
            ]
        completed = set(pd.to_numeric(subset.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        reusable: set[int] = set()
        for _, row in subset.iterrows():
            seed = int(row["seed"])
            checkpoint = resolve_artifact_reference(row.get("best_checkpoint", ""), runtime.paths.stage2_output_search_dirs())
            prediction = resolve_artifact_reference(row.get("prediction_npy", ""), runtime.paths.stage2_output_search_dirs())
            if checkpoint and checkpoint.exists() and prediction and prediction.exists():
                reusable.add(seed)
        rows.append({
            "experiment": name,
            "completed_seeds": len(completed & seeds),
            "artifact_seeds": len(reusable & seeds),
            "required_seeds": len(seeds),
            "complete": seeds.issubset(completed),
            "reusable": seeds.issubset(reusable),
        })
    return pd.DataFrame(rows)



def _completed_rows_for_v13_experiment(runtime: RuntimeConfig, experiment: str) -> pd.DataFrame:
    path = runtime.paths.stage2_file("stage2_v13_results.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    subset = frame[
        (frame.get("experiment", pd.Series(dtype=str)).astype(str) == str(experiment))
        & (frame.get("status", pd.Series(dtype=str)).astype(str) == "completed")
        & (frame.get("split", pd.Series(dtype=str)).astype(str) == V13_SPLIT_NAME)
    ].copy()
    if subset.empty:
        return subset
    keys = [c for c in ("experiment", "split", "seed") if c in subset.columns]
    return subset.drop_duplicates(subset=keys, keep="last")


def lock_v13_winner(
    runtime: RuntimeConfig,
    *,
    selected_experiment: str | None = None,
    candidate_experiments: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Freeze the V13 development winner and its final-training recipe.

    The validation split is used only for development.  The lock stores the median
    best epoch and validity threshold across completed seeds so final training can
    reproduce the same curriculum horizon on 100% of labeled ROIs without looking at
    the hidden challenge test set.
    """
    registry = stage2_experiment_registry()
    names = tuple(candidate_experiments or VERSION13_EXPERIMENTS)
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise KeyError(f"Unknown V13 candidate experiment(s): {unknown}")
    ranking = aggregate_v13_results(runtime, names)
    if ranking.empty:
        raise RuntimeError("Cannot lock V13 Stage 2: no complete results.")

    if selected_experiment is None:
        missing = sorted(set(names) - set(ranking["experiment"].astype(str)))
        if missing:
            raise RuntimeError(
                "Automatic V13 winner selection requires all requested experiments to be complete; "
                f"missing {missing}. Pass selected_experiment explicitly to lock a reviewed winner."
            )
        sort_cols = [
            c for c in ("macro_f1", "conditional_type_macro_f1_present", "reject_f1")
            if c in ranking.columns
        ]
        selected_row = ranking.sort_values(sort_cols, ascending=False, na_position="last").iloc[0]
        selected = str(selected_row["experiment"])
        selection_mode = "v13_balanced_train_val_auto_rank"
    else:
        selected = str(selected_experiment)
        if selected not in registry:
            raise KeyError(f"Unknown selected V13 experiment {selected!r}.")
        if selected not in set(ranking["experiment"].astype(str)):
            raise RuntimeError(f"Selected V13 experiment {selected!r} has no complete development result.")
        selection_mode = "v13_balanced_train_val_manual_review"

    rows = _completed_rows_for_v13_experiment(runtime, selected)
    required_seeds = {int(seed) for seed in runtime.training.seeds}
    observed_seeds = set(pd.to_numeric(rows.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    if not required_seeds.issubset(observed_seeds):
        raise RuntimeError(
            f"Cannot lock {selected}: completed seeds {sorted(observed_seeds)} do not cover "
            f"required seeds {sorted(required_seeds)}."
        )
    best_epochs = pd.to_numeric(rows["best_epoch"], errors="coerce").dropna().astype(int).tolist()
    thresholds = pd.to_numeric(rows["validity_threshold"], errors="coerce").dropna().astype(float).tolist()
    if not best_epochs:
        raise RuntimeError(f"Cannot derive final training epoch from {selected}; best_epoch is missing.")
    recommended_epoch = max(1, int(round(float(np.median(best_epochs)))))
    recommended_threshold = float(np.median(thresholds)) if thresholds else 0.5

    split = ensure_v13_split(runtime, force=False, check_sources=False)
    selected_metrics = ranking[ranking["experiment"].astype(str) == selected].iloc[0].to_dict()
    lock = {
        "version": 13,
        "selected_experiment": selected,
        "selected_at": utc_now_iso(),
        "selection_mode": selection_mode,
        "split_name": V13_SPLIT_NAME,
        "split_hash": split["split_hash"],
        "selected_model_config": asdict(registry[selected]),
        "selected_metrics": selected_metrics,
        "ranking": ranking.to_dict(orient="records"),
        "seeds": [int(seed) for seed in runtime.training.seeds],
        "final_training_plan": {
            "use_all_labeled_rois": True,
            "recommended_epochs": recommended_epoch,
            # Keep curriculum boundaries tied to the development schedule.
            "schedule_reference_epochs": int(runtime.training.epochs),
            "validity_threshold": recommended_threshold,
            "source_best_epochs": [int(v) for v in best_epochs],
            "source_validity_thresholds": [float(v) for v in thresholds],
            "effective_batch_size": int(runtime.training.effective_batch_size),
            "stage2_micro_batch_size": int(runtime.training.stage2_micro_batch_size),
        },
    }
    atomic_write_json(runtime.paths.stage2_file("stage2_v13_lock.json"), lock)
    print(
        f"LOCKED V13 STAGE 2: {selected}; final epoch={recommended_epoch}, "
        f"validity threshold={recommended_threshold:.3f}"
    )
    return lock

def run_stage2_v13_program(
    runtime: RuntimeConfig,
    *,
    hf_token: str | None = None,
    experiments_to_run: Iterable[str] | None = None,
    experiment_group: str | None = None,
    parallel_runs: int = 1,
    lora_parallel_runs: int = 1,
) -> dict[str, Any]:
    names = _resolve_names(experiments_to_run, experiment_group)
    validate_full_oof(runtime)
    split = ensure_v13_split(runtime, force=False, check_sources=False)
    ensure_stage2_pretrained_checkpoints(
        runtime.paths.root, hf_token=hf_token, pfm_keys=("uni2_h",)
    )
    registry = stage2_experiment_registry()
    print("Version-13 Stage-2 experiments requested:")
    for name in names:
        print(f"  - {name}")
    outputs = run_v13_jobs(
        runtime,
        [registry[name] for name in names],
        hf_token=hf_token,
        parallel_runs=parallel_runs,
        lora_parallel_runs=lora_parallel_runs,
    )
    ranking = aggregate_v13_results(runtime, names)
    summary = {
        "version": 13,
        "split": V13_SPLIT_NAME,
        "split_hash": split["split_hash"],
        "experiments": list(names),
        "parallel_runs": int(parallel_runs),
        "lora_parallel_runs": int(lora_parallel_runs),
        "outputs": outputs,
        "aggregate": ranking.to_dict(orient="records"),
    }
    return summary
