from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


@dataclass
class PerformanceMetrics:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)

    def update_val_loss(self, value: float) -> None:
        self.val_loss.append(value)

    def update_train_loss(self, value: float) -> None:
        self.train_loss.append(value)

    def update_train_accuracy(self, value: float) -> None:
        self.train_accuracy.append(value)

    def update_val_accuracy(self, value: float) -> None:
        self.val_accuracy.append(value)

    def plot_training_report(self):
        sns.set_theme()
        is_classifier: bool = len(self.train_accuracy) > 0 or len(self.val_accuracy) > 0

        if is_classifier is True:
            self.__plot_classifier_metrics()
        else:
            self.__plot_regression_metrics()

    def __plot_regression_metrics(self):
        nrows, ncols = 2, 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, layout="constrained")
        ax1, ax2 = axes.flatten()

        ax1.plot_training_report(self.train_loss, 'r')
        ax1.title.set_text('Training loss')
        ax2.plot_training_report(self.val_loss, 'b')
        ax2.title.set_text('Validation loss')

        plt.show()

    def __plot_classifier_metrics(self):
        nrows, ncols = 4, 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, layout="constrained")
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.plot(self.train_loss, 'r')
        ax1.title.set_text('Training loss')
        ax2.plot(self.train_accuracy, 'g')
        ax2.title.set_text('Training accuracy')
        ax3.plot(self.val_loss, 'b')
        ax3.title.set_text('Validation loss')
        ax4.plot(self.val_accuracy, 'k')
        ax4.title.set_text('Validation accuracy')

        plt.show()


def __get_confusion_matrix(predictions, labels):
    labels = labels.detach().cpu().numpy()
    predictions = torch.argmax(predictions, dim=1)
    predictions = predictions.detach().cpu().numpy()
    return confusion_matrix(labels, predictions)


def get_accuracy_score(predictions, labels):
    cm = __get_confusion_matrix(predictions, labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return np.average(cm.diagonal())


def plot_confusion_matrix(predictions, labels):
    cm = __get_confusion_matrix(predictions, labels)

    sns.set_theme()
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%')
    plt.show()
