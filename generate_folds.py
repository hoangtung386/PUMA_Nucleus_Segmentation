from encoder.config import TrainConfig
from encoder.splits import generate_multilabel_folds

if __name__ == '__main__':
    generate_multilabel_folds(TrainConfig(), force=True)
    print('Generated 3-fold multi-label stratified splits.')
