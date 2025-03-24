"""Metrics for segmentation.
"""

import torch
import math
import numpy as np


class Metrics_cls:
    """Tracking mean metrics
    """

    def __init__(self):
        """Creates an new `Metrics` instance.

        Args:
          labels: the labels for all classes.
        """

        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0

    def add(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """
        # print(predicted)
        b = predicted.shape[0]
        for i in range(b):
            cls = 0 if predicted[i][0] > predicted[i][1] else 1  # 预测的类别
            label = actual[i]
            if cls == 0:
                if label == 0:
                    self.tn += 1
                else:
                    self.fn += 1
            else:
                if label == 0:
                    self.fp += 1
                else:
                    self.tp += 1

    def get_precision(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

if __name__ == '__main__':
    predict = torch.tensor(
        [[0, 1],
         [0, 1],
         [0, 1],
         [0, 1]]
    )
    label = torch.tensor([0, 1, 1, 0])
    metrics = Metrics_cls()
    metrics.add(label, predict)
    print(metrics.get_precision())