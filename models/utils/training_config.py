import logging
from dataclasses import dataclass
from typing import Union

import torch
from src.models.utils.data import get_train_val_dataloader


@dataclass
class TrainingConfig:
    train_data: torch.utils.data.DataLoader
    val_data: torch.utils.data.DataLoader
    epochs: int
    device: torch.device
    model: torch.nn.Module
    criterion: torch.nn.modules.loss._Loss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    logger: logging.Logger

    def __repr__(self) -> str:
        return (f'epochs {self.epochs} model {self.model} criterion {self.criterion} optimizer_lr {self.optimizer}'
                f' scheduler: {self.scheduler}').replace('\n', '')


def training_config_builder(dataset_path: str, model: torch.nn.Module, criterion: torch.nn.modules.loss._Loss,
                            optimizer: torch.optim.Optimizer,
                            scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
                            batch_size: int, epochs: int, logger: logging.Logger) -> TrainingConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val = get_train_val_dataloader(dataset_path, batch_size, device)
    logger.info('Initializing data loaders was successful. Train set shape: %s Val set shape: %s',
                train.dataset.tensors[0].size(), val.dataset.tensors[0].size())

    return TrainingConfig(
        train_data=train,
        val_data=val,
        epochs=epochs,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger
    )
