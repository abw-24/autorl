
"""
-- Constructor class for computing symbolic loss for a JSON specified supervised
convolutional neural network. Conventions for specifying layers and options below:
"""

import sys
import tensorflow as tf

from tf_utils import tfUtilities
from utils import *

"""
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
"""


class ConvNet(tfUtilities):

    def __init__(self, x_placeholder, y_placeholder, config_map, log=None):

        self._x = x_placeholder
        self._y = y_placeholder
        self._layers = config_map['layers']
        self._loss_type = config_map['loss']
        self._log = log
        self._loss = None
        self._parameters = []



    def _forward_pass(self):

        op = self._x

        layer_ops_ = {
            "conv": self._conv_layer,
            "fully-connected"
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

    def _conv_layer(self, in_op, param_specs, param_namespace, log=None):


    def _max_pool_layer(self, in_op, param_specs, param_namespace, log=None):
        k = param_specs['strides']
        with param_namespace as name:
            out_op = tf.nn.max_pool(in_op, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                    padding='SAME')
            return out_op

    def _connected_layer(in_op, param_specs, param_namespace, log=None):
        pass

    @property
    def loss(self):

        if self._loss is None:
            forward_op = self._forward_pass()
            self._loss = self.compute_loss(forward_op, self._y, self._loss_type)

        return self._loss