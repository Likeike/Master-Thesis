from typing import Tuple

import torch

from src.models.loss_func import cross_entropy_loss
from src.models.utils.metrics import PerformanceMetrics, get_accuracy_score
from src.models.utils.training_config import TrainingConfig


def train(config: TrainingConfig) -> Tuple[PerformanceMetrics, torch.nn.Module]:
    _model = config.model
    _scheduler = config.scheduler
    _criterion = config.criterion
    _device = config.device
    _logger = config.logger
    performance_metrics = PerformanceMetrics([], [], [], [])

    _logger.info('Starting model training with config: %s', config)

    for epoch in range(config.epochs):

        train_loss_per_epoch, val_loss_per_epoch = 0, 0
        n_batches = len(config.train_data)

        # train loop
        for i, batch in enumerate(config.train_data):
            _model.train()
            batch = [r.to(_device) for r in batch]
            xs, ys = batch
            config.optimizer.zero_grad()
            ys_hat = _model(xs)

            loss = cross_entropy_loss(_criterion, ys_hat, ys)
            loss.backward()
            config.optimizer.step()

            if _scheduler:
                _scheduler.step()

        train_loss_per_epoch = train_loss_per_epoch + loss.item()
        performance_metrics.update_train_loss(train_loss_per_epoch / n_batches)
        train_acc = get_accuracy_score(ys_hat, ys)
        performance_metrics.update_train_accuracy(train_acc)

        # val loop
        for i, batch in enumerate(config.val_data):
            _model.eval()
            batch = [r.to(_device) for r in batch]
            xs, ys = batch
            with torch.no_grad():
                ys_hat = _model(xs)
                loss = cross_entropy_loss(_criterion, ys_hat, ys)

        val_loss_per_epoch = val_loss_per_epoch + loss.item()
        performance_metrics.update_val_loss(val_loss_per_epoch / n_batches)
        val_acc = get_accuracy_score(ys_hat, ys)
        performance_metrics.update_val_accuracy(val_acc)

        _logger.info('Epoch #%d\ttrain loss: %f\tval loss: %f\ttrain accuracy: %f\tval accuracy: %f',
                     epoch, train_loss_per_epoch, val_loss_per_epoch, train_acc, val_acc)

    return performance_metrics, _model
