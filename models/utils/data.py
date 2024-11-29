from os import path

import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_train_val_dataloader(filepath: str, batch_size: int = 64, device: torch.device = torch.device("cpu")) -> (
        torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    samples, labels = _get_data(filepath, device)
    train_samples, val_samples, train_labels, val_labels = train_test_split(samples, labels, random_state=69420,
                                                                            shuffle=True)
    return _init_dataloaders(train_samples, train_labels, val_samples, val_labels, batch_size=batch_size)


def get_dataset(filepath: str, device: torch.device = torch.device("cpu")) -> TensorDataset:
    return _get_data(filepath, device)


def get_dataset_path(dataset_name: str) -> str:
    dataset_relative_path = '../../data'
    return path.join(path.abspath(path.curdir), dataset_relative_path, dataset_name)


def _init_dataloaders(x_train, y_train, x_val, y_val, batch_size) -> (
        torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    train_data = TensorDataset(x_train, torch.tensor(y_train.tolist()))
    val_data = TensorDataset(x_val, torch.tensor(y_val.tolist()))
    train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, sampler=None, batch_size=batch_size)

    return train_dataloader, val_dataloader


def _get_data(filepath: str, device: torch.device) -> TensorDataset:
    """
    Read the training data and convert to tensors
    """
    tensor_dataset = torch.load(f'{filepath}', device)

    return tensor_dataset[:]
