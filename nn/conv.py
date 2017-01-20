
"""
-- Class for computing symbolic loss for an arbitrary depth conv net
"""

import tensorflow as tf
from tf_utilities import *


class convNet():

    def __init__(self, config_map):

        self._layers = config_map['layers']
        self._loss = None
        self._parameters = []

    def _build_graph(self):

        _layer_ops = {
            "conv": self._conv_layer,
            "relu": tf.nn.relu_layer,
            "max-pool":self._max_pool_layer
        }

        for index, layer in enumerate(self._layers):

            layer_type = layer['layer_type']
            layer_specs = layer['specs']

    @property
    def loss(self):

        if self._loss is None:
            self._compute_loss()

        return self._loss

    @staticmethod
    def _conv_layer(in_op, param_specs):
        pass

    @staticmethod
    def _max_pool_layer(in_op, param_specs):
        pass