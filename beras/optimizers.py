from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        for p, grad in zip(trainable_params, grads):
            val = p - self.learning_rate * grad
            p.assign(val)

class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        for i, (p, grad) in enumerate(zip(trainable_params, grads)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad ** 2)

            step = (self.learning_rate / (np.sqrt(self.v[i]) + self.epsilon)) * grad
            val = p - step

            p.assign(val)


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params, grads):
        self.t += 1
        for i, (p, grad) in enumerate(zip(trainable_params, grads)):
            self.m[i] = self.m[i] * self.beta_1 + (1 - self.beta_1) * grad
            self.v[i] = self.v[i] * self.beta_2 + (1 - self.beta_2) * (grad ** 2)

            m = self.m[i] / (1 - self.beta_1 ** self.t)
            v = self.v[i] / (1 - self.beta_2 ** self.t)

            new_param = p - (self.learning_rate * m) / (np.sqrt(v) + self.epsilon)
            p.assign(new_param)
