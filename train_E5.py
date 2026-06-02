from encoder.engine import run_by_name

if __name__ == '__main__':
    # Default: train all 3 folds for this experiment.
    # To train only one fold, use: python train.py --run E5 --fold 0
    run_by_name('E5', all_folds=True)
