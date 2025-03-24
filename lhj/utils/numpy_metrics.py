import numpy as np
import pydensecrf.densecrf as dcrf
import cv2
from cv2 import imread, imwrite

def binary_classification_metrics(predictions, labels):
    # 将预测图和标签二值化为 0 或 1
    predictions_binary = predictions.astype(int)
    labels_binary = labels.astype(int)

    # 计算 True Positives、False Positives、False Negatives
    TP = np.sum(np.logical_and(predictions_binary == 255, labels_binary == 255))
    FP = np.sum(np.logical_and(predictions_binary == 255, labels_binary == 0))
    FN = np.sum(np.logical_and(predictions_binary == 0, labels_binary == 255))

    # # 计算 Precision、Recall 和 F1
    # precision = TP / (TP + FP) if TP + FP > 0 else 0
    # recall = TP / (TP + FN) if TP + FN > 0 else 0
    # f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # return precision, recall, f1
    return TP, FP, FN