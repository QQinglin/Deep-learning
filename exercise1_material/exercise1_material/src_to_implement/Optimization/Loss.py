from audioop import error
import numpy as np
import torch
from fontTools.misc.bezierTools import epsilon
from scipy.ndimage import label
from torch.nn.functional import one_hot


class CrossEntropyLoss:
    def __init__(self):
        self.softmax_pred = None  # Store Softmax predictions
        self.prediction_tensor =None

    def forward(self, prediction_tensor, label_tensor):
        slices = label_tensor == 1
        epsilon = np.finfo(np.float64).eps  # To prevent log(0)
        self.prediction_tensor = prediction_tensor.copy()
        loss = np.sum(-np.log(self.prediction_tensor[slices] + epsilon) ) # Cross-entropy loss
        return loss

    def backward(self, label_tensor):
        grad = -label_tensor/self.prediction_tensor  # Compute gradient
        return grad