#!/usr/bin/env python3
"""Create the fixed, class-aware Stage-2 train/validation split.

The split is written separately from the five Stage-1 OOF folds. Case IDs are
respected when available; otherwise grouping falls back to ROI IDs. The optimizer
balances ROI count, nuclei count, all ten classes, class presence, rare classes,
and primary/metastatic composition.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PUMA_CLASS_NAMES: tuple[str, ...] = (
    "nuclei_tumor",
    "nuclei_lymphocyte",
    "nuclei_plasma_cell",
    "nuclei_histiocyte",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_stroma",
    "nuclei_endothelium",
    "nuclei_epithelium",
    "nuclei_apoptosis",
)
PUMA_CLASS_TO_ID = {name: i for i, name in enumerate(PUMA_CLASS_NAMES)}
# Rare classes receive stronger split-balance weights.
TAIL_CLASS_NAMES: tuple[str, ...] = (
    "nuclei_plasma_cell",
    "nuclei_neutrophil",
    "nuclei_apoptosis",
    "nuclei_melanophage",
    "nuclei_endothelium",
)
TAIL_CLASS_IDS = tuple(PUMA_CLASS_TO_ID[name] for name in TAIL_CLASS_NAMES)

SPLIT_TRAIN = np.int8(0)
SPLIT_VAL = np.int8(1)
SPLIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    image_dir: Path
    geojson_dir: Path
    artifact_dir: Path
    manifest_file: Path
    case_metadata_csv: Path
    output_dir: Path

    @classmethod
    def from_root(cls, root: Path, output_dir: Path | None = None) -> "ProjectPaths":
        root = Path(root).expanduser().resolve()
        artifact_dir = root / "PUMA_outputs"
        return cls(
            root=root,
            image_dir=root / "Dataset" / "01_training_dataset_tif_ROIs",
            geojson_dir=root / "Dataset" / "01_training_dataset_geojson_nuclei",
            artifact_dir=artifact_dir,
            manifest_file=artifact_dir / "puma_roi_manifest.npy",
            case_metadata_csv=root / "Dataset" / "puma_case_metadata.csv",
            output_dir=(Path(output_dir).expanduser().resolve() if output_dir else artifact_dir / "train_val_split"),
        )


@dataclass
class GroupRecord:
    case_id: str
    roi_indices: np.ndarray
    roi_count: int
    total_nuclei: int
    class_counts: np.ndarray
    class_roi_presence: np.ndarray
    melanoma_roi_counts: np.ndarray  # [primary, metastatic, other]


@dataclass
class MetricSpec:
    name: str
    values: np.ndarray
    target: float
    scale: float
    weight: float


@dataclass
class SplitResult:
    val_group_mask: np.ndarray
    objective: float
    optimizer: str
    status: str
    message: str


def _required_manifest_fields() -> set[str]:
    return {
        "roi_index",
        "roi_id",
        "image_file",
        "geojson_file",
        "melanoma_type",
        "case_id",
        "n_nuclei",
        *(f"count_class_{i}" for i in range(len(PUMA_CLASS_NAMES))),
    }


def load_and_validate_manifest(paths: ProjectPaths, *, check_sources: bool = True) -> np.ndarray:
    if not paths.manifest_file.exists():
        raise FileNotFoundError(
            f"Missing preprocessing manifest: {paths.manifest_file}\n"
            "Run 00_Preprocess.ipynb first. This split generator intentionally reuses "
            "the class counts stored by preprocessing."
        )
    manifest = np.load(paths.manifest_file, allow_pickle=False)
    if manifest.dtype.names is None:
        raise TypeError(f"Expected a structured NPY manifest, got dtype={manifest.dtype}")
    missing = sorted(_required_manifest_fields() - set(manifest.dtype.names))
    if missing:
        raise ValueError(
            "puma_roi_manifest.npy is missing required fields: "
            + ", ".join(missing)
        )
    n = len(manifest)
    if n < 10:
        raise ValueError(f"Too few ROIs for a stable train/validation split: {n}")
    roi_indices = np.asarray(manifest["roi_index"], dtype=np.int64)
    if not np.array_equal(np.sort(roi_indices), np.arange(n, dtype=np.int64)):
        raise ValueError("roi_index must contain exactly 0..N-1 so split assignments align with NPY artifacts.")
    if check_sources:
        if not paths.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {paths.image_dir}")
        if not paths.geojson_dir.is_dir():
            raise FileNotFoundError(f"GeoJSON directory does not exist: {paths.geojson_dir}")
        missing_images: list[str] = []
        missing_geojson: list[str] = []
        for row in manifest:
            image_file = paths.image_dir / str(row["image_file"])
            geojson_file = paths.geojson_dir / str(row["geojson_file"])
            if not image_file.is_file():
                missing_images.append(str(image_file))
            if not geojson_file.is_file():
                missing_geojson.append(str(geojson_file))
            if len(missing_images) >= 10 and len(missing_geojson) >= 10:
                break
        if missing_images or missing_geojson:
            raise FileNotFoundError(
                "The manifest is not aligned with the configured raw input paths. "
                f"Missing images (first 10): {missing_images[:10]}; "
                f"missing GeoJSONs (first 10): {missing_geojson[:10]}."
            )
    class_counts = np.column_stack(
        [np.asarray(manifest[f"count_class_{i}"], dtype=np.int64) for i in range(len(PUMA_CLASS_NAMES))]
    )
    if np.any(class_counts < 0):
        raise ValueError("Negative class counts detected in manifest.")
    n_nuclei = np.asarray(manifest["n_nuclei"], dtype=np.int64)
    if not np.array_equal(class_counts.sum(axis=1), n_nuclei):
        bad = np.flatnonzero(class_counts.sum(axis=1) != n_nuclei)[:10]
        raise ValueError(f"Manifest class counts do not sum to n_nuclei for ROI indices {bad.tolist()}.")
    return manifest


def build_case_groups(manifest: np.ndarray) -> list[GroupRecord]:
    grouped: dict[str, list[int]] = {}
    for row_position, row in enumerate(manifest):
        case_id = str(row["case_id"]).strip() or str(row["roi_id"]).strip()
        grouped.setdefault(case_id, []).append(row_position)

    groups: list[GroupRecord] = []
    for case_id in sorted(grouped):
        rows = np.asarray(grouped[case_id], dtype=np.int64)
        class_counts = np.zeros(len(PUMA_CLASS_NAMES), dtype=np.int64)
        class_roi_presence = np.zeros(len(PUMA_CLASS_NAMES), dtype=np.int64)
        melanoma = np.zeros(3, dtype=np.int64)
        for idx in rows:
            row = manifest[idx]
            counts = np.asarray(
                [int(row[f"count_class_{i}"]) for i in range(len(PUMA_CLASS_NAMES))],
                dtype=np.int64,
            )
            class_counts += counts
            class_roi_presence += (counts > 0).astype(np.int64)
            melanoma_type = str(row["melanoma_type"]).strip().lower()
            if melanoma_type == "primary":
                melanoma[0] += 1
            elif melanoma_type == "metastatic":
                melanoma[1] += 1
            else:
                melanoma[2] += 1
        groups.append(
            GroupRecord(
                case_id=case_id,
                roi_indices=np.asarray(manifest["roi_index"][rows], dtype=np.int64),
                roi_count=int(len(rows)),
                total_nuclei=int(class_counts.sum()),
                class_counts=class_counts,
                class_roi_presence=class_roi_presence,
                melanoma_roi_counts=melanoma,
            )
        )
    if len(groups) < 2:
        raise ValueError("Need at least two distinct case groups for train/validation splitting.")
    return groups


def _rare_weight(total_count: float, max_count: float, class_id: int) -> float:
    # 1 for the most common class; approaches ~4 for very rare classes.
    ratio = max(float(total_count), 1.0) / max(float(max_count), 1.0)
    inverse_rarity = float(np.clip(1.0 / math.sqrt(ratio), 1.0, 6.0))
    weight = 1.0 + 0.6 * (inverse_rarity - 1.0)
    if class_id in TAIL_CLASS_IDS:
        weight *= 2.0
    return weight


def build_metric_specs(groups: list[GroupRecord], val_fraction: float) -> list[MetricSpec]:
    g = len(groups)
    roi_counts = np.asarray([r.roi_count for r in groups], dtype=np.float64)
    nuclei = np.asarray([r.total_nuclei for r in groups], dtype=np.float64)
    class_counts = np.stack([r.class_counts for r in groups]).astype(np.float64)
    class_presence = np.stack([r.class_roi_presence for r in groups]).astype(np.float64)
    melanoma = np.stack([r.melanoma_roi_counts for r in groups]).astype(np.float64)

    specs: list[MetricSpec] = []

    def add(name: str, values: np.ndarray, weight: float) -> None:
        total = float(values.sum())
        target = total * val_fraction
        # The objective uses normalized absolute error. Keep the denominator meaningful
        # for small rare-class targets without allowing one nucleus to dominate everything.
        scale = max(target, 1.0)
        specs.append(MetricSpec(name=name, values=values, target=target, scale=scale, weight=weight))

    add("roi_count", roi_counts, 18.0)
    add("case_group_count", np.ones(g, dtype=np.float64), 6.0)
    add("total_nuclei", nuclei, 3.0)

    max_class_count = max(float(class_counts.sum(axis=0).max()), 1.0)
    for class_id, name in enumerate(PUMA_CLASS_NAMES):
        rw = _rare_weight(class_counts[:, class_id].sum(), max_class_count, class_id)
        add(f"class_count::{name}", class_counts[:, class_id], 5.0 * rw)
        # ROI-presence balance is especially important for rare classes: it avoids a
        # seemingly balanced count where all validation examples come from one image.
        add(f"class_roi_presence::{name}", class_presence[:, class_id], 8.0 * rw)

    melanoma_names = ("primary", "metastatic", "other")
    for j, name in enumerate(melanoma_names):
        if melanoma[:, j].sum() > 0:
            add(f"melanoma_roi_count::{name}", melanoma[:, j], 3.0)
    return specs


def _coverage_bounds(groups: list[GroupRecord], val_fraction: float) -> list[tuple[np.ndarray, float, float, str]]:
    """Return (values, lower, upper, label) constraints on validation assignment.

    Values are per-group counts of ROIs containing a class. Bounds are selected only
    when feasible. Tail classes receive a stronger minimum of two validation source
    ROIs when at least eight source ROIs exist across at least four case groups.
    """
    class_presence = np.stack([r.class_roi_presence for r in groups]).astype(np.float64)
    group_binary = (class_presence > 0).astype(np.float64)
    bounds: list[tuple[np.ndarray, float, float, str]] = []
    for class_id, name in enumerate(PUMA_CLASS_NAMES):
        total_roi_presence = int(class_presence[:, class_id].sum())
        total_group_presence = int(group_binary[:, class_id].sum())
        if total_group_presence < 2:
            # Impossible to put this class in both sets without leaking a case.
            continue
        min_val = 1
        min_train = 1
        if class_id in TAIL_CLASS_IDS and total_roi_presence >= 8 and total_group_presence >= 4:
            min_val = min(2, max(1, int(round(total_roi_presence * val_fraction))))
            min_train = 2
        # validation presence <= total presence - minimum train presence
        upper = float(total_roi_presence - min_train)
        if min_val <= upper:
            bounds.append((class_presence[:, class_id], float(min_val), upper, f"coverage::{name}"))
    return bounds


def solve_milp(
    groups: list[GroupRecord],
    specs: list[MetricSpec],
    *,
    val_fraction: float,
    roi_fraction_tolerance: float,
    time_limit_seconds: float,
) -> SplitResult:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:  # pragma: no cover - fallback is tested separately by behavior
        raise RuntimeError(f"scipy.optimize.milp is unavailable: {exc}") from exc

    n_groups = len(groups)
    n_metrics = len(specs)
    # Variables: x_g (binary val assignment), dplus_m, dminus_m.
    n_vars = n_groups + 2 * n_metrics
    c = np.zeros(n_vars, dtype=np.float64)
    integrality = np.zeros(n_vars, dtype=np.int8)
    integrality[:n_groups] = 1
    lower = np.zeros(n_vars, dtype=np.float64)
    upper = np.full(n_vars, np.inf, dtype=np.float64)
    upper[:n_groups] = 1.0

    for m, spec in enumerate(specs):
        coefficient = spec.weight / max(spec.scale, 1e-12)
        c[n_groups + 2 * m] = coefficient
        c[n_groups + 2 * m + 1] = coefficient

    rows: list[tuple[np.ndarray, float, float]] = []
    # Equality for each absolute-deviation metric:
    # sum(values*x) - dplus + dminus == target
    for m, spec in enumerate(specs):
        coeff = np.zeros(n_vars, dtype=np.float64)
        coeff[:n_groups] = spec.values
        coeff[n_groups + 2 * m] = -1.0
        coeff[n_groups + 2 * m + 1] = 1.0
        rows.append((coeff, spec.target, spec.target))

    # Tight but feasible ROI-count band around the requested fraction.
    roi_values = np.asarray([r.roi_count for r in groups], dtype=np.float64)
    total_rois = float(roi_values.sum())
    target_rois = total_rois * val_fraction
    tol_rois = max(1.0, total_rois * roi_fraction_tolerance)
    min_rois = max(1.0, math.floor(target_rois - tol_rois))
    max_rois = min(total_rois - 1.0, math.ceil(target_rois + tol_rois))
    coeff = np.zeros(n_vars, dtype=np.float64)
    coeff[:n_groups] = roi_values
    rows.append((coeff, min_rois, max_rois))

    # Class coverage constraints.
    for values, lb, ub, _label in _coverage_bounds(groups, val_fraction):
        coeff = np.zeros(n_vars, dtype=np.float64)
        coeff[:n_groups] = values
        rows.append((coeff, lb, ub))

    A = lil_matrix((len(rows), n_vars), dtype=np.float64)
    lb = np.empty(len(rows), dtype=np.float64)
    ub = np.empty(len(rows), dtype=np.float64)
    for i, (coeff, lo, hi) in enumerate(rows):
        nz = np.flatnonzero(coeff)
        A.rows[i] = nz.tolist()
        A.data[i] = coeff[nz].tolist()
        lb[i] = lo
        ub[i] = hi

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options={"time_limit": float(time_limit_seconds), "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status}; message={result.message}")
    mask = np.asarray(result.x[:n_groups] >= 0.5, dtype=bool)
    if mask.all() or (~mask).all():
        raise RuntimeError("MILP returned a degenerate all-train or all-validation split.")
    return SplitResult(
        val_group_mask=mask,
        objective=float(result.fun if result.fun is not None else np.nan),
        optimizer="scipy.optimize.milp",
        status=str(result.status),
        message=str(result.message),
    )


def _objective_for_mask(mask: np.ndarray, specs: list[MetricSpec]) -> float:
    total = 0.0
    for spec in specs:
        actual = float(spec.values[mask].sum())
        total += spec.weight * abs(actual - spec.target) / max(spec.scale, 1e-12)
    return float(total)


def solve_fallback(
    groups: list[GroupRecord],
    specs: list[MetricSpec],
    *,
    val_fraction: float,
    seed: int,
    restarts: int = 100,
) -> SplitResult:
    """Deterministic multi-start greedy/local search fallback.

    The fallback includes strong feasibility penalties for coverage. The MILP path is
    preferred because it gives a much cleaner optimum under the defined linear objective.
    """
    rng = np.random.default_rng(seed)
    n = len(groups)
    roi_values = np.asarray([g.roi_count for g in groups], dtype=np.int64)
    target_rois = int(round(roi_values.sum() * val_fraction))
    coverage = _coverage_bounds(groups, val_fraction)

    def penalized(mask: np.ndarray) -> float:
        if mask.all() or (~mask).all():
            return 1e12
        score = _objective_for_mask(mask, specs)
        val_rois = int(roi_values[mask].sum())
        score += 25.0 * abs(val_rois - target_rois) / max(target_rois, 1)
        for values, lb, ub, _ in coverage:
            actual = float(values[mask].sum())
            if actual < lb:
                score += 1e6 * (lb - actual)
            if actual > ub:
                score += 1e6 * (actual - ub)
        return float(score)

    best_mask: np.ndarray | None = None
    best_score = float("inf")
    for restart in range(max(1, int(restarts))):
        order = rng.permutation(n)
        mask = np.zeros(n, dtype=bool)
        running = 0
        for idx in order:
            if running < target_rois:
                mask[idx] = True
                running += int(roi_values[idx])
        # Hill climb with random flips/swaps.
        score = penalized(mask)
        for _ in range(max(1000, n * 30)):
            proposal = mask.copy()
            if rng.random() < 0.55:
                i = int(rng.integers(0, n))
                proposal[i] = ~proposal[i]
            else:
                val_ids = np.flatnonzero(proposal)
                train_ids = np.flatnonzero(~proposal)
                if len(val_ids) == 0 or len(train_ids) == 0:
                    continue
                i = int(rng.choice(val_ids))
                j = int(rng.choice(train_ids))
                proposal[i] = False
                proposal[j] = True
            pscore = penalized(proposal)
            if pscore + 1e-12 < score:
                mask, score = proposal, pscore
        if score < best_score:
            best_score = score
            best_mask = mask.copy()
    if best_mask is None or best_score >= 1e6:
        raise RuntimeError("Fallback optimizer could not find a feasible split.")
    return SplitResult(
        val_group_mask=best_mask,
        objective=float(best_score),
        optimizer="deterministic_greedy_local_search",
        status="fallback",
        message="MILP unavailable/failed; used deterministic multi-start local search.",
    )


def optimize_split(
    groups: list[GroupRecord],
    *,
    val_fraction: float,
    roi_fraction_tolerance: float,
    seed: int,
    time_limit_seconds: float,
) -> tuple[SplitResult, list[MetricSpec]]:
    if not (0.05 <= val_fraction <= 0.40):
        raise ValueError("val_fraction should be between 0.05 and 0.40 for this development split.")
    if not (0.0 <= roi_fraction_tolerance <= 0.10):
        raise ValueError("roi_fraction_tolerance must be between 0 and 0.10.")
    specs = build_metric_specs(groups, val_fraction)
    try:
        result = solve_milp(
            groups,
            specs,
            val_fraction=val_fraction,
            roi_fraction_tolerance=roi_fraction_tolerance,
            time_limit_seconds=time_limit_seconds,
        )
    except Exception as exc:
        print(f"[split] MILP optimizer unavailable/failed ({exc}); using fallback.", file=sys.stderr)
        result = solve_fallback(
            groups,
            specs,
            val_fraction=val_fraction,
            seed=seed,
        )
    return result, specs


def group_mask_to_roi_assignments(
    groups: list[GroupRecord], val_group_mask: np.ndarray, n_rois: int
) -> np.ndarray:
    assignments = np.full(n_rois, -1, dtype=np.int8)
    for group, is_val in zip(groups, val_group_mask, strict=True):
        assignments[group.roi_indices] = SPLIT_VAL if is_val else SPLIT_TRAIN
    if np.any(assignments < 0):
        raise AssertionError("Some ROI indices were not assigned to train/validation.")
    if not np.any(assignments == SPLIT_TRAIN) or not np.any(assignments == SPLIT_VAL):
        raise AssertionError("Degenerate split: train or validation is empty.")
    return assignments


def _summary_dict(manifest: np.ndarray, assignments: np.ndarray, split_value: int) -> dict[str, Any]:
    mask = assignments == split_value
    rows = manifest[mask]
    counts = np.asarray(
        [[int(row[f"count_class_{i}"]) for i in range(len(PUMA_CLASS_NAMES))] for row in rows],
        dtype=np.int64,
    )
    if len(rows) == 0:
        counts = np.zeros((0, len(PUMA_CLASS_NAMES)), dtype=np.int64)
    class_counts = counts.sum(axis=0) if len(counts) else np.zeros(len(PUMA_CLASS_NAMES), dtype=np.int64)
    presence = (counts > 0).sum(axis=0) if len(counts) else np.zeros(len(PUMA_CLASS_NAMES), dtype=np.int64)
    case_ids = {str(row["case_id"]).strip() or str(row["roi_id"]) for row in rows}
    melanoma = {
        "primary": int(np.sum(np.asarray([str(x).lower() == "primary" for x in rows["melanoma_type"]], dtype=bool))),
        "metastatic": int(np.sum(np.asarray([str(x).lower() == "metastatic" for x in rows["melanoma_type"]], dtype=bool))),
    }
    melanoma["other"] = int(len(rows) - melanoma["primary"] - melanoma["metastatic"])
    return {
        "roi_count": int(len(rows)),
        "case_group_count": int(len(case_ids)),
        "nuclei_count": int(class_counts.sum()),
        "class_counts": class_counts.astype(int).tolist(),
        "class_roi_presence": presence.astype(int).tolist(),
        "melanoma_roi_counts": melanoma,
    }


def validate_final_split(manifest: np.ndarray, assignments: np.ndarray) -> dict[str, Any]:
    train = _summary_dict(manifest, assignments, int(SPLIT_TRAIN))
    val = _summary_dict(manifest, assignments, int(SPLIT_VAL))

    # Case leakage check.
    case_to_splits: dict[str, set[int]] = {}
    for row, split in zip(manifest, assignments, strict=True):
        case_id = str(row["case_id"]).strip() or str(row["roi_id"])
        case_to_splits.setdefault(case_id, set()).add(int(split))
    leaked = sorted(case for case, splits in case_to_splits.items() if len(splits) > 1)
    if leaked:
        raise AssertionError(f"Case leakage detected: {leaked[:10]}")

    class_counts_total = np.asarray(train["class_counts"]) + np.asarray(val["class_counts"])
    class_presence_total = np.asarray(train["class_roi_presence"]) + np.asarray(val["class_roi_presence"])
    missing_val = [PUMA_CLASS_NAMES[i] for i, v in enumerate(val["class_counts"]) if int(v) == 0 and class_counts_total[i] > 0]
    missing_train = [PUMA_CLASS_NAMES[i] for i, v in enumerate(train["class_counts"]) if int(v) == 0 and class_counts_total[i] > 0]
    return {
        "train": train,
        "validation": val,
        "case_leakage_count": 0,
        "missing_validation_classes": missing_val,
        "missing_train_classes": missing_train,
        "total_class_roi_presence": class_presence_total.astype(int).tolist(),
    }


def write_outputs(
    paths: ProjectPaths,
    manifest: np.ndarray,
    assignments: np.ndarray,
    *,
    result: SplitResult,
    specs: list[MetricSpec],
    val_fraction: float,
    roi_fraction_tolerance: float,
    seed: int,
    check_sources: bool,
) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = paths.output_dir / "puma_train_val_assignments.npy"
    indices_path = paths.output_dir / "puma_train_val_indices.npz"
    split_csv = paths.output_dir / "puma_train_val_split.csv"
    summary_csv = paths.output_dir / "puma_train_val_class_summary.csv"
    metadata_json = paths.output_dir / "puma_train_val_split_metadata.json"

    np.save(assignments_path, assignments.astype(np.int8), allow_pickle=False)
    np.savez_compressed(
        indices_path,
        train_roi_indices=np.flatnonzero(assignments == SPLIT_TRAIN).astype(np.int64),
        val_roi_indices=np.flatnonzero(assignments == SPLIT_VAL).astype(np.int64),
    )

    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "roi_index", "roi_id", "case_id", "melanoma_type", "split", "n_nuclei",
            *[f"count_{name}" for name in PUMA_CLASS_NAMES],
        ]
        writer.writerow(header)
        for row, split in zip(manifest, assignments, strict=True):
            writer.writerow([
                int(row["roi_index"]),
                str(row["roi_id"]),
                str(row["case_id"]),
                str(row["melanoma_type"]),
                "validation" if int(split) == int(SPLIT_VAL) else "train",
                int(row["n_nuclei"]),
                *[int(row[f"count_class_{i}"]) for i in range(len(PUMA_CLASS_NAMES))],
            ])

    diagnostics = validate_final_split(manifest, assignments)
    train = diagnostics["train"]
    val = diagnostics["validation"]
    total_class_counts = np.asarray(train["class_counts"], dtype=np.int64) + np.asarray(val["class_counts"], dtype=np.int64)
    total_presence = np.asarray(train["class_roi_presence"], dtype=np.int64) + np.asarray(val["class_roi_presence"], dtype=np.int64)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_id", "class_name", "is_tail_class",
            "total_nuclei", "train_nuclei", "val_nuclei", "val_nuclei_fraction",
            "total_roi_presence", "train_roi_presence", "val_roi_presence", "val_presence_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, name in enumerate(PUMA_CLASS_NAMES):
            total_n = int(total_class_counts[i])
            total_p = int(total_presence[i])
            writer.writerow({
                "class_id": i,
                "class_name": name,
                "is_tail_class": int(i in TAIL_CLASS_IDS),
                "total_nuclei": total_n,
                "train_nuclei": int(train["class_counts"][i]),
                "val_nuclei": int(val["class_counts"][i]),
                "val_nuclei_fraction": (float(val["class_counts"][i]) / total_n if total_n else ""),
                "total_roi_presence": total_p,
                "train_roi_presence": int(train["class_roi_presence"][i]),
                "val_roi_presence": int(val["class_roi_presence"][i]),
                "val_presence_fraction": (float(val["class_roi_presence"][i]) / total_p if total_p else ""),
            })

    roi_ids = [str(row["roi_id"]) for row in manifest]
    unique_cases = {str(row["case_id"]).strip() or str(row["roi_id"]) for row in manifest}
    all_case_ids_equal_roi_ids = all(
        (str(row["case_id"]).strip() or str(row["roi_id"])) == str(row["roi_id"]) for row in manifest
    )
    metric_diagnostics = []
    val_group_dummy = None  # only for readability below
    for spec in specs:
        # Reconstruct actual from ROI assignment for the metrics we can expose; the raw
        # objective is already saved, so this list documents target/importance.
        metric_diagnostics.append({
            "name": spec.name,
            "target": spec.target,
            "weight": spec.weight,
            "normalization_scale": spec.scale,
        })

    metadata = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "project_root": str(paths.root),
        "input_paths": {
            "image_dir": str(paths.image_dir),
            "geojson_dir": str(paths.geojson_dir),
            "manifest_file": str(paths.manifest_file),
            "case_metadata_csv": str(paths.case_metadata_csv),
        },
        "source_path_check_enabled": bool(check_sources),
        "requested_validation_fraction": float(val_fraction),
        "roi_fraction_tolerance": float(roi_fraction_tolerance),
        "random_seed": int(seed),
        "optimizer": result.optimizer,
        "optimizer_status": result.status,
        "optimizer_message": result.message,
        "weighted_objective": float(result.objective),
        "number_of_rois": int(len(manifest)),
        "number_of_case_groups": int(len(unique_cases)),
        "case_metadata_file_present": bool(paths.case_metadata_csv.is_file()),
        "case_id_is_roi_id_for_all_rows": bool(all_case_ids_equal_roi_ids),
        "train": train,
        "validation": val,
        "actual_validation_roi_fraction": float(val["roi_count"] / len(manifest)),
        "actual_validation_nuclei_fraction": float(val["nuclei_count"] / max(train["nuclei_count"] + val["nuclei_count"], 1)),
        "case_leakage_count": diagnostics["case_leakage_count"],
        "missing_validation_classes": diagnostics["missing_validation_classes"],
        "missing_train_classes": diagnostics["missing_train_classes"],
        "class_names": list(PUMA_CLASS_NAMES),
        "tail_class_names": list(TAIL_CLASS_NAMES),
        "metric_objectives": metric_diagnostics,
        "notes": [
            "Stage-1 fold assignments remain separate from the Stage-2 development split.",
            "Use puma_train_val_indices.npz or puma_train_val_assignments.npy in the next Stage-2 training implementation.",
            "If case_id equals roi_id because patient metadata is unavailable, grouping is ROI-level by necessity.",
            "For the final Grand Challenge model, freeze hyperparameters on this split, then retrain on 100% of labeled data.",
        ],
    }
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "assignments": assignments_path,
        "indices": indices_path,
        "split_csv": split_csv,
        "summary_csv": summary_csv,
        "metadata": metadata_json,
    }


def print_summary(manifest: np.ndarray, assignments: np.ndarray, outputs: dict[str, Path]) -> None:
    diagnostics = validate_final_split(manifest, assignments)
    tr, va = diagnostics["train"], diagnostics["validation"]
    total_rois = tr["roi_count"] + va["roi_count"]
    total_nuclei = tr["nuclei_count"] + va["nuclei_count"]
    print("\n=== PUMA optimized train/validation split ===")
    print(f"ROIs:    train={tr['roi_count']}  val={va['roi_count']}  val_fraction={va['roi_count']/total_rois:.4f}")
    print(f"Cases:   train={tr['case_group_count']}  val={va['case_group_count']}")
    print(f"Nuclei:  train={tr['nuclei_count']}  val={va['nuclei_count']}  val_fraction={va['nuclei_count']/max(total_nuclei,1):.4f}")
    print(f"Leakage: case_leakage_count={diagnostics['case_leakage_count']}")
    if diagnostics["missing_validation_classes"]:
        print("WARNING - classes missing from validation:", diagnostics["missing_validation_classes"])
    if diagnostics["missing_train_classes"]:
        print("WARNING - classes missing from train:", diagnostics["missing_train_classes"])
    print("\nPer-class validation ratios (nuclei / ROI-presence):")
    total_counts = np.asarray(tr["class_counts"]) + np.asarray(va["class_counts"])
    total_presence = np.asarray(tr["class_roi_presence"]) + np.asarray(va["class_roi_presence"])
    for i, name in enumerate(PUMA_CLASS_NAMES):
        nc = int(total_counts[i])
        pc = int(total_presence[i])
        nr = (va["class_counts"][i] / nc) if nc else float("nan")
        pr = (va["class_roi_presence"][i] / pc) if pc else float("nan")
        tail = " [TAIL]" if i in TAIL_CLASS_IDS else ""
        print(f"  {name:<25}{tail:<7} nuclei={nr:6.2%}  ROI_presence={pr:6.2%}  val_n={va['class_counts'][i]}")
    print("\nSaved:")
    for key, path in outputs.items():
        print(f"  {key:>12}: {path}")


def create_split(
    project_dir: Path | str,
    *,
    val_fraction: float = 0.20,
    roi_fraction_tolerance: float = 0.02,
    seed: int = 2026,
    time_limit_seconds: float = 120.0,
    output_dir: Path | str | None = None,
    check_sources: bool = True,
) -> dict[str, Path]:
    paths = ProjectPaths.from_root(Path(project_dir), Path(output_dir) if output_dir else None)
    manifest = load_and_validate_manifest(paths, check_sources=check_sources)
    groups = build_case_groups(manifest)
    result, specs = optimize_split(
        groups,
        val_fraction=val_fraction,
        roi_fraction_tolerance=roi_fraction_tolerance,
        seed=seed,
        time_limit_seconds=time_limit_seconds,
    )
    assignments = group_mask_to_roi_assignments(groups, result.val_group_mask, len(manifest))
    outputs = write_outputs(
        paths,
        manifest,
        assignments,
        result=result,
        specs=specs,
        val_fraction=val_fraction,
        roi_fraction_tolerance=roi_fraction_tolerance,
        seed=seed,
        check_sources=check_sources,
    )
    print(f"Optimizer: {result.optimizer}; objective={result.objective:.6f}; status={result.status}")
    print_summary(manifest, assignments, outputs)
    return outputs
