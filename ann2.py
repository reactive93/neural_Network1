# encoding: utf-8
import numpy as np
from numpy import ndarray
from typing import List

from loss import SquareLoss
from optimizers2 import SGS
from node2 import Node
from utils import batch_iterator
from copy import copy
import pickle
import sys

from utils import accuracy, acc_test

class Net(object):

    def __init__(self,optimizer=SGS()):
        self.layers:List[Node]=[]
        self.loss_function = SquareLoss()
        self.optimizer = optimizer
    def add(self, node:Node):
        self.layers.append(node)

    def compile(self):
        prev_out=0
        for node in self.layers:
            if node.inputs is not None:
                node.compile()
                node.set_optimizer( copy(self.optimizer))
                prev_out = node.neurons
            else:
                node.set_optimizer(copy(self.optimizer))
                node.compile2(prev_out)
                prev_out = node.neurons

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._feedforward(X)
        loss = np.mean(self.loss_function.loss(y, y_pred))
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward(loss_grad=loss_grad)
        acc =accuracy(y,y_pred) #accuracy_score(y,y_pred)
        return loss,acc

    def fit(self,X:ndarray,Y:ndarray,epoch:int=None,batch_size:int=None, test_acc=()):

        loss=0
        local_loss=0
        acc_local=0
        acc=0
        epoch_iter=0
        for _ in range(epoch):

            for X_batch, y_batch in batch_iterator(X, Y, batch_size=batch_size):
                local_loss, acc_local = self.train_on_batch(X_batch, y_batch)
                loss+=local_loss
                acc=acc_local

            epoch_iter+=1
                # print('local loss: ',local_loss)
            print('epoch: ',epoch_iter, 'global loss:',loss,' accuracy: ',acc)
            loss=0
            acc=0

            if len(test_acc)>1:

                predict_test = self._feedforward(test_acc[0])
                acc_test = accuracy(test_acc[1],predict_test)
                print('test acc: {0}'.format(acc_test) )



    def _feedforward(self,X:ndarray):
        out_put = X

        for layer in self.layers:

            out_put = layer.forward(out_put)

        return out_put


    def _backward(self,loss_grad):

        acc_grad = loss_grad

        for layer in reversed(self.layers):
            acc_grad = layer.backward(acc_grad)

    def predict(self,X:ndarray):

        predict = self._feedforward(X)
        return predict


    def save_model(self,path:str=None):
        path1 = path
        if path1 is None:
            path1 = sys.path[0]+'/model.pickle'

        with open(path1,'wb') as file:

            pickle.dump(self.layers,file)

    def load_model(self,path:str):

        with open(path, 'rb') as file:

            self.layers = pickle.load(file)
