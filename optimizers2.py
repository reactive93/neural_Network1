# encoding: utf-8
from numpy import ndarray,dot,transpose
import numpy as np
class SGS():

    def __init__(self,lr=0.01,momentum=0):
        self.lr = lr
        self.momentum = momentum


    def update(self, w:ndarray, grad_w:ndarray, inputs:ndarray):


        delta = dot(transpose(inputs), grad_w)
        w_upd = self.momentum * delta + (1-delta) * self.lr * delta


        return w + w_upd


class Adagrad():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None # Sum of squares of the gradients
        self.eps = 1e-8

    def update(self, w:ndarray, grad_w:ndarray, inputs:ndarray):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        delta = dot(transpose(inputs), grad_w)
        self.G += np.power(delta, 2)


        w_upd = delta + (1 - delta) * self.learning_rate * delta


        return w + self.learning_rate * w_upd / np.sqrt(self.G + self.eps)


class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w:ndarray, grad_w:ndarray, inputs:ndarray):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(w))

        delta = dot(transpose(inputs), grad_w)
        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(delta, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w + self.learning_rate * delta / np.sqrt(self.Eg + self.eps)


class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w:ndarray, grad_w:ndarray, inputs:ndarray):
        # If not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(w))
            self.v = np.zeros(np.shape(w))

        delta = dot(transpose(inputs), grad_w)

        self.m = self.b1 * self.m + (1 - self.b1) * delta
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(delta, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


        return w + w_updt