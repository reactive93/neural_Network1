# encoding: utf-8
import numpy as np
from numpy import exp,dot,transpose,ndarray,array
from activation_functions import Sigmoid

class Node(object):

    def __init__(self,neurons:int,inputs:int=None,activation=Sigmoid()):
        self.weight:np.ndarray = None
        self._activationFunc = activation
        self.neurons = neurons
        self.inputs = inputs
        self.linear=None
        self.nonlinear:np.ndarray = None
        self.nonlinearDx:np.ndarray = None
        self.error:np.ndarray = None
        self.delta_error:np.ndarray = None
        self.layer_input:ndarray = None
        self.optimizer = None
        pass


    def set_optimizer(self,optimizer):
        self.optimizer = optimizer


    def compile(self):
        self.weight: np.ndarray = np.random.uniform(1.48, 1.7 ** -6.2, (self.inputs, self.neurons ))* np.sqrt(2.0/(self.neurons))

    def compile2(self,inputs:int):
        self.weight: np.ndarray = np.random.uniform(1.48, 1.7 ** -6.2, (inputs, self.neurons))* np.sqrt(2.0/(self.neurons))





    def forward(self, X:ndarray):
        self.layer_input = X
        self.linear = dot( X, self.weight )
        self.nonlinear = self._activationFunc( self.linear )
        return self.nonlinear

    def backward(self, error:ndarray):

        W = self.weight

        self.delta_error = error * self._activationFunc.gradient(self.linear)

        self.weight = self.optimizer.update(self.weight,self.delta_error,self.layer_input)
        next_err = dot(error,transpose(W))

        return next_err