import logging

import torch.nn as nn
import torch.optim as optim

from model import Net
from utils.metrics import get_accuracy_score, plot_confusion_matrix
from utils.data import get_dataset_path
from utils.training_config import training_config_builder
from utils.data import get_dataset
from train import train
from loss_func import cross_entropy_loss

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def main() -> None:
    model = Net()
    #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_MULT)

    try:
        training_config = training_config_builder(
            dataset_path=get_dataset_path('pairs_train-2024-11-16-50mels-13mfcc.pt'), model=model, criterion=nn.CrossEntropyLoss(),
            optimizer=optim.SGD(params=model.parameters(), lr=0.01, momentum=.9), scheduler=None, batch_size=64,
            epochs=15, logger=logger)
    except Exception as e:
        logger.exception('An uncaught exception occurred while initializing the training: %s', e)
        return

    try:
        training_performance_metrics, model = train(training_config)
    except Exception as e:
        logger.exception('An uncaught exception occurred during training: %s', e)
        return

    training_performance_metrics.plot_training_report()

    predict(model)


def predict(model):
    test_data, labels = get_dataset(get_dataset_path('pairs_test-2024-11-16-50mels-13mfcc.pt'))

    predictions = model.forward(test_data)
    loss = cross_entropy_loss(nn.CrossEntropyLoss(), predictions, labels).item()
    accuracy = get_accuracy_score(predictions, labels)
    logger.info('Test data performance:\t loss: %f\taccuracy:%f', loss, accuracy)
    plot_confusion_matrix(predictions, labels)


if __name__ == '__main__':
    main()
