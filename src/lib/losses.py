import sys
sys.path.append("../")
from config import Config
import numpy as np

class Loss:
    def __init__(self, config: Config):
        self.config = config
        self.regularization = config.trainer.regularization
    
    def mean_square_error(self, y_pred, y_true, params):
        loss = np.mean(np.sum(np.square(y_pred - y_true), 1)) * 0.5
        return (loss, loss+self.regularization * 0.5 * (np.mean(np.sum(np.square(params[0]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[1]))) + np.mean(np.sum(np.square(params[2]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[3]))) + np.mean(np.sum(np.square(params[4]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[5]))) + np.mean(np.sum(np.square(params[6]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[7])))))

    def cross_entropy(self, y_pred, y_true, params):
        loss = -np.mean(np.sum(y_true * np.log((y_pred + 1e-8).astype(np.float32)), 1))
        return (loss, loss+self.regularization * 0.5 * (np.mean(np.sum(np.square(params[0]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[1]))) + np.mean(np.sum(np.square(params[2]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[3]))) + np.mean(np.sum(np.square(params[4]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[5]))) + np.mean(np.sum(np.square(params[6]), 1)) + np.mean(np.einsum("ijk->i", np.square(params[7])))))
    
    def get(self, loss_type):
        all_loss = {
            "mse": self.mean_square_error,
            "crossentropy": self.cross_entropy
        }
        if loss_type.lower() in all_loss:
            loss_type = loss_type.lower()
            return all_loss[loss_type]
        else:
            return all_loss["mse"]


def accuracy(y_pred, y_true):
    if len(y_true[0]) == 1:
        return np.mean(np.equal(np.where(y_pred<0.5, 0, 1), y_true).astype(np.float32))
    else:
        return np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(y_true, 1)).astype(np.float32))