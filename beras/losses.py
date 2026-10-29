import numpy as np

from beras.core import Diffable, Tensor



class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        squared_diff = (y_pred - y_true) ** 2
        return np.mean(squared_diff)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        n = y_pred.size
        predicted_y_gradient = 2 * (y_pred - y_true) / n
        true_y_gradient = np.zeros_like(y_true)
        
        return [predicted_y_gradient, true_y_gradient]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss_per_sample = -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
        
        return np.mean(loss_per_sample)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        batch_size = y_pred.shape[0]
        eps = 1e-15

        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        predicted_y_gradient = - (y_true / y_pred_clipped) / batch_size
        true_y_gradient = np.zeros_like(y_true)
        
        return [predicted_y_gradient, true_y_gradient]