import cv2
import torch
import math
import numpy as np


class Metrics:
    def __init__(self, labels):

        self.labels = labels

        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0

    def add(self, label, predict):
        masks = torch.argmax(predict, 0)
        ground_truths = label
        pred = masks

        # ground_truths = label.cpu()
        # pred = predict

        self.tn += np.array(torch.sum((ground_truths == 0) & (pred == 0)).cpu())
        self.fn += np.array(torch.sum((ground_truths == 1) & (pred == 0)).cpu())
        self.fp += np.array(torch.sum((ground_truths == 0) & (pred == 1)).cpu())
        self.tp += np.array(torch.sum((ground_truths == 1) & (pred == 1)).cpu())

    def get_precision(self):

        return self.tp / (self.tp + self.fp)

    def get_recall(self):

        return self.tp / (self.tp + self.fn)

    def get_f_score(self):

        pr = 2 * (self.tp / (self.tp + self.fp)) * self.tp / (self.tp + self.fn)
        p_r = (self.tp / (self.tp + self.fp)) + (self.tp / (self.tp + self.fn))

        return pr / p_r

    def get_oa(self):

        t_pn = self.tp + self.tn
        t_tpn = self.tp + self.tn + self.fp + self.fn
        return t_pn / t_tpn

    def get_miou(self):

        return np.nanmean(self.tp / (self.tp + self.fn + self.fp))
        # return np.nanmean([self.tn / (self.tn + self.fn + self.fp), self.tp / (self.tp + self.fn + self.fp)])