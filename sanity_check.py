import argparse

from encoder.sanity import run_sanity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', action='store_true', help='Also instantiate each model and run a small synthetic forward pass.')
    parser.add_argument('--full-size', action='store_true', help='Use full 1024x1024 model forward. Memory-heavy.')
    parser.add_argument('--force-folds', action='store_true', help='Regenerate the 3-fold split before checking.')
    args = parser.parse_args()
    run_sanity(model_forward=args.models, full_size_forward=args.full_size, force_folds=args.force_folds)
