import numpy as np
import tensorflow.keras as keras
import math

class BaseDNN(object):
    def __init__(self):
        super().__init__()
        self.units = []
    
    def call(self, inputs):
        """Forward pass on the layer"""
        Z = inputs
        for layer in self.units:
            Z = layer(Z)
        return Z

    def __add__(self, layer):
        if layer:
            self.units.append(layer)
        return self


class DNNLayer(BaseDNN, keras.layers.Layer):
    def __init__(self):
        super().__init__()


class DNNModel(BaseDNN, keras.models.Model):
    def __init__(self):
        super().__init__()



class EarlyStopper(object):
    def __init__(self, patience):
        self.patience = patience
        self.best_metric = np.NINF
        self.epochs_not_improving = 0

    def should_stop(self, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            self.epochs_not_improving = 0
        else:
            self.epochs_not_improving += 1
        return self.epochs_not_improving == self.patience

class DropLearningRate(object):
    def __init__(self, drop_rate, epoch_drop_rate, lr0):
        self.drop_rate = drop_rate
        self.lr0 = lr0
        self.epoch_drop_rate = epoch_drop_rate

    def __call__(self, epoch):
        return self.lr0 * math.pow(self.drop_rate, math.floor(epoch / self.epoch_drop_rate))
