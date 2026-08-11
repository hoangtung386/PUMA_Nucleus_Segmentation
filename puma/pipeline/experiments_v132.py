from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from puma.config import RuntimeConfig
from puma.models.stage2 import ensure_stage2_pretrained_checkpoints
from puma.pipeline.oof import validate_full_oof
from puma.stage2.catalog import (
    VERSION132_EXPERIMENTS,
    stage2_experiment_groups,
    stage2_experiment_registry,
)
from puma.stage2.runner_v132 import run_v132_jobs
from puma.training.stage2_v132 import V132_SPLIT_NAME, ensure_v132_split
from puma.training.stage2_v132 import _run_hash
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
        group = experiment_group or "screening"
        groups = stage2_experiment_groups()
        if group not in groups:
            raise KeyError(f"Unknown V13.2 experiment group {group!r}: {sorted(groups)}")
        names = groups[group]
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise KeyError(f"Unknown V13.2 Stage-2 experiment(s): {unknown}")
    if not names:
        raise ValueError("At least one V13.2 Stage-2 experiment must be selected.")
    return names


def aggregate_v132_results(
    runtime: RuntimeConfig,
    experiments: Iterable[str] | None = None,
    *,
    epoch_profile: int | None = None,
) -> pd.DataFrame:
    path = runtime.paths.stage2_file("stage2_v132_results.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    keys = [c for c in ("experiment", "split", "seed", "epoch_profile") if c in frame.columns]
    frame = frame.drop_duplicates(subset=keys, keep="last")
    frame = frame[
        (frame["status"].astype(str) == "completed")
        & (frame["split"].astype(str) == V132_SPLIT_NAME)
    ].copy()
    profile = int(epoch_profile or runtime.training.stage2_epochs)
    if "epoch_profile" in frame:
        frame = frame[pd.to_numeric(frame["epoch_profile"], errors="coerce") == profile]
    if experiments is not None:
        frame = frame[frame["experiment"].astype(str).isin(tuple(experiments))]
    if frame.empty:
        return frame

    registry = stage2_experiment_registry()
    split = ensure_v132_split(runtime, force=False, check_sources=False)
    current_split_hash = str(split["split_hash"])
    keep = []
    for idx_row, row in frame.iterrows():
        name = str(row.get("experiment", ""))
        if name in registry and str(row.get("config_hash", "")) == _run_hash(runtime, registry[name], current_split_hash):
            keep.append(idx_row)
    frame = frame.loc[keep].copy()
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(
        subset=["experiment", "split", "seed", "epoch_profile", "config_hash"], keep="last"
    )

    required_seeds = {int(seed) for seed in runtime.training.seeds}
    complete_names: list[str] = []
    for name, rows in frame.groupby("experiment"):
        observed = set(pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int).tolist())
        if required_seeds.issubset(observed):
            complete_names.append(str(name))
    frame = frame[frame["experiment"].astype(str).isin(complete_names)]
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
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    aggregate = frame.groupby("experiment", as_index=False)[metric_columns].mean(numeric_only=True)
    if {"reject_precision", "reject_recall"}.issubset(aggregate.columns):
        p = aggregate["reject_precision"].fillna(0.0)
        r = aggregate["reject_recall"].fillna(0.0)
        aggregate["reject_f1"] = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return aggregate


def lock_v132_winner(
    runtime: RuntimeConfig,
    *,
    selected_experiment: str | None = None,
    candidate_experiments: Iterable[str] | None = None,
) -> dict[str, Any]:
    registry = stage2_experiment_registry()
    names = tuple(candidate_experiments or VERSION132_EXPERIMENTS)
    ranking = aggregate_v132_results(runtime, names)
    if ranking.empty:
        raise RuntimeError("Cannot lock V13.2 Stage 2: no complete results for this epoch profile.")
    if selected_experiment is None:
        missing = sorted(set(names) - set(ranking["experiment"].astype(str)))
        if missing:
            raise RuntimeError(
                "Automatic V13.2 selection requires all requested experiments complete; "
                f"missing {missing}."
            )
        sort_cols = [
            c for c in ("macro_f1", "conditional_type_macro_f1_present", "reject_f1")
            if c in ranking.columns
        ]
        selected = str(ranking.sort_values(sort_cols, ascending=False, na_position="last").iloc[0]["experiment"])
        selection_mode = "v132_auto_rank"
    else:
        selected = str(selected_experiment)
        if selected not in registry:
            raise KeyError(f"Unknown V13.2 experiment {selected!r}.")
        selection_mode = "v132_manual_review"

    path = runtime.paths.stage2_file("stage2_v132_results.csv")
    frame = pd.read_csv(path)
    current_split = ensure_v132_split(runtime, force=False, check_sources=False)
    selected_config_hash = _run_hash(runtime, registry[selected], str(current_split["split_hash"]))
    rows = frame[
        (frame["experiment"].astype(str) == selected)
        & (frame["status"].astype(str) == "completed")
        & (frame["split"].astype(str) == V132_SPLIT_NAME)
        & (pd.to_numeric(frame["epoch_profile"], errors="coerce") == int(runtime.training.stage2_epochs))
        & (frame["config_hash"].astype(str) == selected_config_hash)
    ].copy()
    rows = rows.drop_duplicates(subset=["experiment", "split", "seed", "epoch_profile", "config_hash"], keep="last")
    required = {int(seed) for seed in runtime.training.seeds}
    observed = set(pd.to_numeric(rows["seed"], errors="coerce").dropna().astype(int).tolist())
    if not required.issubset(observed):
        raise RuntimeError(f"Completed seeds {sorted(observed)} do not cover required {sorted(required)}.")
    thresholds = pd.to_numeric(rows["validity_threshold"], errors="coerce").dropna().astype(float).tolist()
    best_epochs = pd.to_numeric(rows["best_epoch"], errors="coerce").dropna().astype(int).tolist()
    split = ensure_v132_split(runtime, force=False, check_sources=False)
    lock = {
        "version": "13.2",
        "selected_experiment": selected,
        "selected_at": utc_now_iso(),
        "selection_mode": selection_mode,
        "epoch_profile": int(runtime.training.stage2_epochs),
        "split_name": V132_SPLIT_NAME,
        "split_hash": split["split_hash"],
        "selected_model_config": asdict(registry[selected]),
        "selected_config_hash": selected_config_hash,
        "selected_metrics": ranking[ranking["experiment"].astype(str) == selected].iloc[0].to_dict(),
        "ranking": ranking.to_dict(orient="records"),
        "seeds": sorted(required),
        "deployment": {
            "validity_threshold": float(np.median(thresholds)) if thresholds else 0.5,
            "source_best_epochs": [int(v) for v in best_epochs],
            "recommended_final_epochs": 100,
        },
    }
    atomic_write_json(runtime.paths.stage2_file("stage2_v132_lock.json"), lock)
    print(
        f"LOCKED V13.2: {selected}; development profile={runtime.training.stage2_epochs}ep; "
        f"deployment threshold={lock['deployment']['validity_threshold']:.3f}."
    )
    return lock


def run_stage2_v132_program(
    runtime: RuntimeConfig,
    *,
    hf_token: str | None = None,
    experiments_to_run: Iterable[str] | None = None,
    experiment_group: str | None = None,
) -> dict[str, Any]:
    names = _resolve_names(experiments_to_run, experiment_group)
    validate_full_oof(runtime)
    split = ensure_v132_split(runtime, force=False, check_sources=False)
    ensure_stage2_pretrained_checkpoints(
        runtime.paths.root, hf_token=hf_token, pfm_keys=("uni2_h",)
    )
    registry = stage2_experiment_registry()
    print(f"V13.2 Stage-2 profile: {runtime.training.stage2_epochs} epochs")
    for name in names:
        print(f"  - {name}")
    outputs = run_v132_jobs(
        runtime,
        [registry[name] for name in names],
        hf_token=hf_token,
    )
    ranking = aggregate_v132_results(runtime, names)
    return {
        "version": "13.2",
        "epoch_profile": int(runtime.training.stage2_epochs),
        "split": V132_SPLIT_NAME,
        "split_hash": split["split_hash"],
        "experiments": list(names),
        "outputs": outputs,
        "aggregate": ranking.to_dict(orient="records"),
    }
