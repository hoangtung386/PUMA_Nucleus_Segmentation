from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from encoder.config import ENCODER_RUNS, OUTPUT_DIR


OFFICIAL_METRIC = 'official_selection_score'
OLD_METRIC = 'selection_score'


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def summarize() -> Path:
    rows = []

    for key, cfg in ENCODER_RUNS.items():
        for fold_id in range(cfg.n_folds):
            metrics_path = OUTPUT_DIR / cfg.experiment_name / f'fold_{fold_id}' / 'metrics.csv'
            if not metrics_path.exists():
                rows.append({
                    'run': key,
                    'experiment': cfg.experiment_name,
                    'fold': fold_id,
                    'status': 'missing',
                })
                continue

            df = pd.read_csv(metrics_path)
            if df.empty or OFFICIAL_METRIC not in df.columns:
                rows.append({
                    'run': key,
                    'experiment': cfg.experiment_name,
                    'fold': fold_id,
                    'status': 'empty_or_missing_official_selection_score',
                })
                continue

            best_idx = df[OFFICIAL_METRIC].idxmax()
            best = df.loc[best_idx].to_dict()
            rows.append({
                'run': key,
                'experiment': cfg.experiment_name,
                'fold': fold_id,
                'status': 'ok',
                'best_epoch': int(best.get('epoch', -1)),
                'official_selection_score': _safe_float(best.get('official_selection_score')),
                'nuclei_f1': _safe_float(best.get('nuclei_f1')),
                'tissue_dice': _safe_float(best.get('tissue_dice')),
                'nuclei_macro_f1': _safe_float(best.get('nuclei_macro_f1')),
                'tissue_micro_dice': _safe_float(best.get('tissue_micro_dice')),
                # Keep legacy fields if present.
                'selection_score': _safe_float(best.get('selection_score')),
                'mean_tissue_dice_scored_1_to_5': _safe_float(best.get('mean_tissue_dice_scored_1_to_5')),
                'dice_nuclei_foreground': _safe_float(best.get('dice_nuclei_foreground')),
                'mean_nuclei_class_dice_on_nuclei_pixels': _safe_float(best.get('mean_nuclei_class_dice_on_nuclei_pixels')),
            })

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'comparison_summary_by_fold.csv'
    summary.to_csv(out_path, index=False)

    ok = summary[summary['status'] == 'ok'].copy()
    if not ok.empty:
        # Best fold for each experiment = fold with highest official score.
        best_fold_idx = ok.groupby(['run', 'experiment'])['official_selection_score'].idxmax()
        best_fold_df = ok.loc[best_fold_idx, [
            'run',
            'experiment',
            'fold',
            'best_epoch',
            'nuclei_f1',
            'tissue_dice',
            'nuclei_macro_f1',
            'tissue_micro_dice',
            'official_selection_score',
        ]].rename(columns={
            'fold': 'best_fold',
            'best_epoch': 'best_epoch_of_best_fold',
            'nuclei_f1': 'best_fold_nuclei_f1',
            'tissue_dice': 'best_fold_tissue_dice',
            'nuclei_macro_f1': 'best_fold_nuclei_macro_f1',
            'tissue_micro_dice': 'best_fold_tissue_micro_dice',
            'official_selection_score': 'best_fold_official_selection_score',
        })

        agg = ok.groupby(['run', 'experiment'], as_index=False).agg(
            folds_done=('fold', 'count'),
            official_selection_score_mean=('official_selection_score', 'mean'),
            official_selection_score_std=('official_selection_score', 'std'),
            nuclei_f1_mean=('nuclei_f1', 'mean'),
            tissue_dice_mean=('tissue_dice', 'mean'),
            nuclei_macro_f1_mean=('nuclei_macro_f1', 'mean'),
            tissue_micro_dice_mean=('tissue_micro_dice', 'mean'),
            # Legacy means kept for continuity with older result files.
            selection_score_mean=('selection_score', 'mean'),
            selection_score_std=('selection_score', 'std'),
        )

        agg = agg.merge(best_fold_df, on=['run', 'experiment'], how='left')
        agg = agg.sort_values('official_selection_score_mean', ascending=False)

        agg_path = OUTPUT_DIR / 'comparison_summary_mean_over_folds.csv'
        agg.to_csv(agg_path, index=False)
        print(agg.to_string(index=False))
        print(f'Wrote {agg_path}')

    print(f'Wrote {out_path}')
    return out_path


if __name__ == '__main__':
    summarize()
