

"""
-- Constructor class for computing symbolic loss for an arbitrary network.
Provides a caffe-like interface to Tensorflow in the sense that layers
 are JSON configurable. Conventions for layer specifications can be
 found in the README (under construction)

 # "layers"
 # "options"

"""


import tensorflow as tf
from tf_utils import tfUtilities
import numpy as np

import math


class LayerCompute(object):

    def __init__(self, layer_type):

        pass

    def _input_layer(self, layer_op, previous_dim):

        assert previous_dim is None, "Input layer specified in layer past the first. Exiting."

    def _dense_hidden_layer(self, layer_op, previous_dim):
        pass

    def _convolution_layer(self, layer_op, previous_dim):
        raise NotImplementedError('Conv nets not yet available')

    def _pooling_layer(self, layer_op, previous_dim):
        raise NotImplementedError('Conv nets not yet available')

    def _output_layer(self, layer_op, previous_dim):
        pass

    def compute(self, layer_op, previous_dim):
        pass


class NN(tfUtilities):

    def __init__(self, x_placeholder, y_placeholder, model_ops, weights=None, biases=None, wait=False):
        """
        Builds the computatuional graph for a multi-layer perception

        :param input_holder:
        :param input_dim:
        :param model_ops:
        :return:
        """

        self.x_placeholder = x_placeholder
        self.y_placeholder = y_placeholder

        self._layers = model_ops['layers']
        self._ops = model_ops['options']

        # parse global architecture options
        self._loss_type = self._ops['loss_type']

        # set up instance objects to be filled with later
        self._loss = None
        self._y_hat = None
        self.inWeights = weights
        self.inBiases = biases
        self._weights = []
        self._biases = []

        if not wait:
            self.build()

    def _layer_build(self):
        """

        :return: hidden tensor
        """

        layer_compute = None

        previous_dim = None

        # build weight matrices and bias vectors for each specified layer
        for index, layer_map in enumerate(self._layers):

            i_scope = 'layer_' + str(index)

            _type = layer_map['type']
            _dim = layer_map['dim']
            _act = layer_map['activation']

            with tf.name_scope(i_scope):

                if self.inWeights is None:
                    # using truncated normals for weight initialization, zeros for biases
                    weights = tf.Variable(tf.truncated_normal([row_dim, col_dim],
                                                              stddev=1.0 / math.sqrt(float(row_dim))),
                                          name='weights')
                else:
                    weights = tf.Variable(self.inWeights[index].astype(np.float32), name='weights')

                self._weights.append(weights)

                if self.inBiases is None:
                    biases = tf.Variable(tf.zeros([col_dim]), name='biases')
                else:
                    biases = tf.Variable(self.inBiases[index].astype(np.float32), name='biases')

                self._biases.append(biases)

                previous_dim = _dim
                previous_type = _type

                linear_op = tf.matmul(layer_compute if layer_compute is not None else self.x_placeholder, weights)
                linear_w_biases = linear_op + biases
                non_linear_op = self.activate(linear_w_biases, h)
                layer_compute = non_linear_op

        return layer_compute

    def build(self):
        """

        :return:
        """

        symbolic_hidden_transformation = self._hidden_layers()
        self._y_hat = self._output_layer(symbolic_hidden_transformation)

        labels = self.y_placeholder
        batch_loss = self.compute_loss(self._y_hat, labels, self._loss_type)
        self._loss = tf.reduce_mean(batch_loss, name=self._loss_type)

    @property
    def loss(self):

        if self._loss is None:
            self.build()

        return self._loss

    @property
    def y_hat(self):

        if self._y_hat is None:
            self.build()

        return self._y_hat

    @property
    def weights(self):

        if len(self._weights) == 0:
            self.build()

        return self._weights

    @property
    def biases(self):

        if len(self._biases) == 0:
            self.build()

        return self._biases