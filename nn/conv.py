
"""
-- Constructor class for computing symbolic loss for a JSON specified supervised
convolutional neural network. Conventions for specifying layers and options below:
"""

import tensorflow as tf

from tf_utilities import tfUtilities
from utils import *


class convNet(tfUtilities):

    def __init__(self, x_placeholder, y_placeholder, config_map):

        self._x = x_placeholder
        self._y = y_placeholder
        self._layers = config_map['layers']
        self._loss_type = config_map['loss']
        self._loss = None
        self._parameters = []

        self._warning = 'Not all parameters have been set for this layer. Using default.'
        self._exception = 'Unrecognized keyword.'
        self._except_exit = lambda x: 'Missing required parameter: {}'.format(x)

    def _forward_pass(self):

        op = self._x

        layer_ops_ = {
            "conv": self._conv_layer,
            "relu": self._relu_layer,
            "max-pool":self._max_pool_layer
        }

        for index, layer in enumerate(self._layers):

            # pop off the layer type, and send the rest of the layer
            # dictionary as specs to the layer operator
            layer_type = layer.pop('layer_type')
            operator_ = layer_ops_[layer_type]
            layer_namespace = "layer_" + str(index)

            op = operator_(op, layer, layer_namespace)

        return op

    @staticmethod
    def _conv_layer(in_op, param_specs, param_namespace):
        pass

    @staticmethod
    def _max_pool_layer(in_op, param_specs, param_namespace):
        pass

    @staticmethod
    def _relu_layer(in_op, param_specs, param_namespace):
        pass

    @property
    def loss(self):

        if self._loss is None:
            forward_op = self._forward_pass()
            self._loss = self.compute_loss(forward_op, self._y, self._loss_type)

        return self._loss