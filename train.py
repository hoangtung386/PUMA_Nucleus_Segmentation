import argparse

from encoder.config import ENCODER_RUNS
from encoder.engine import run_by_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, choices=sorted(ENCODER_RUNS.keys()))
    parser.add_argument('--fold', type=int, default=None, help='Fold id. If omitted, trains all 3 folds.')
    parser.add_argument('--all-folds', action='store_true', help='Train all folds explicitly.')
    args = parser.parse_args()
    run_by_name(args.run, fold=args.fold, all_folds=args.all_folds or args.fold is None)
