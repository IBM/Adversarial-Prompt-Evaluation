"""
Utilities for training and evalutation metrics
"""

from typing import Tuple, Optional
import os

import torch
import matplotlib
import numpy as np

from sklearn import metrics
from matplotlib import pyplot as plt
from ape.metrics.metric import Metric


class MetricComputations(Metric):
    """
    Helper class to handle the metrics tracked for the classifier
    """

    def __init__(self) -> None:

        super().__init__()
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0

        self.epoch_f1 = []
        self.epoch_acc = []
        self.epoch_loss = []

    def compute(self, labels, outputs, loss=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a running F1 score and accuracy
        :param labels:
        :param outputs:
        :param loss:
        """
        labels = labels.detach().cpu().numpy()
        if isinstance(outputs, torch.Tensor):
            # Assumes already argmaxed
            preds = outputs.detach().cpu().numpy()
        else:
            preds = torch.argmax(outputs.logits, axis=1).detach().cpu().numpy()

        self.tp += np.sum(np.logical_and(preds, labels))
        self.tn += np.sum((np.logical_and(np.where(preds == 0, 1, 0), np.where(labels == 0, 1, 0))))

        self.fp += np.sum((np.logical_and(preds, np.where(labels == 0, 1, 0))))
        self.fn += np.sum(np.logical_and(np.where(preds == 0, 1, 0), labels))

        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        if 2 * self.tp + self.fn + self.fp > 0:
            f1 = 2 * self.tp / (2 * self.tp + self.fn + self.fp)
        else:
            f1 = 0

        self.epoch_f1.append(f1)
        self.epoch_acc.append(acc)
        if loss is not None:
            self.epoch_loss.append(loss.data.detach().cpu().numpy())

        return self.epoch_acc[-1], self.epoch_f1[-1], np.mean(self.epoch_loss)
