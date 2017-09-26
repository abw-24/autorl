

"""
-- Constructor class for computing symbolic loss for an arbitrary network.
Provides a caffe-like interface to Tensorflow in the sense that layers
 are JSON configurable. Conventions for layer specifications can be
 found in the README (under construction)

"""


"""
-- MLP in TensorFlow
"""

import tensorflow as tf
from tf_utils import tfUtilities
import numpy as np

import math


class MLP(tfUtilities):

    def __init__(self, x_placeholder, y_placeholder, feature_dim, model_ops, weights=None, biases=None):
        """
        Builds the computatuional graph for a multi-layer perception

        :param input_holder:
        :param input_dim:
        :param model_ops:
        :return:
        """

        self.x_placeholder = x_placeholder
        self.y_placeholder = y_placeholder
        self._feature_dim = feature_dim
        self._n_layers = model_ops['n_layers']
        self._hidden_units = model_ops['n_units']
        self._activation_type = model_ops['activation']
        self.n_output_nodes = model_ops['output_nodes']
        self._loss_type = model_ops['loss_type']

        self._loss = None
        self._y_hat = None
        self.inWeights = weights
        self.inBiases = biases
        self._weights = []
        self._biases = []

        # handling for layer specification.
        # the number of units and activation function
        # specification can either be passed as a list
        # (values for each layer), or can be passed
        # as single values (with the assumption that
        # these values are to be repeated for each layer)

        if isinstance(self._hidden_units, int):
            self._hidden_units = [self._hidden_units] * self._n_layers
        else:
            assert len(self._hidden_units) == self._n_layers

        if isinstance(self._activation_type, str) or isinstance(self._activation_type, unicode):
            self._activation_type = [self._activation_type] * self._n_layers
        else:
            assert len(self._activation_type) == self._n_layers

    def _hidden_layers(self):
        """

        :return: hidden tensor
        """

        layer_compute = None

        # build weight matrices and bias vectors for each specified layer
        for index, ops in enumerate(zip(self._hidden_units, self._activation_type)):

            col_dim, h = ops

            i_scope = 'hidden_' + str(index)

            if index == 0:
                row_dim = self._feature_dim
            else:
                row_dim = self._hidden_units[index - 1]

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

                linear_op = tf.matmul(layer_compute if layer_compute is not None else self.x_placeholder, weights)
                linear_w_biases = linear_op + biases
                non_linear_op = self.activate(linear_w_biases, h)
                layer_compute = non_linear_op

        return layer_compute

    def _output_layer(self, symbolic_hidden_tensor):
        """

        :param symbolic_hidden_tensor:
        :return:
        """

        with tf.name_scope('output_layer'):

            x_dim = self._hidden_units[-1]

            if self.inWeights is None:
                # using truncated normals for weight initialization, zeros for biases
                weights = tf.Variable(tf.truncated_normal([x_dim, self.n_output_nodes],
                                                          stddev=1.0 / math.sqrt(float(x_dim))),
                                      name='weights')
            else:
                weights = tf.Variable(self.inWeights[-1].astype(np.float32), name='weights')

            self._weights.append(weights)

            if self.inBiases is None:
                biases = tf.Variable(tf.zeros([self.n_output_nodes]), name='biases')
            else:
                biases = tf.Variable(self.inBiases[-1].astype(np.float32), name='biases')

            self._biases.append(biases)

            out = tf.matmul(symbolic_hidden_tensor, weights) + biases

        return out

    def symbolic_loss(self):
        """

        :return:
        """

        symbolic_hidden_transformation = self._hidden_layers()
        self._y_hat = self._output_layer(symbolic_hidden_transformation)

        labels = self.y_placeholder
        batch_loss = utils.compute_loss(self._y_hat, labels, self._loss_type)
        self._loss = tf.reduce_mean(batch_loss, name=self._loss_type)

    @property
    def loss(self):

        if self._loss is None:
            self.symbolic_loss()

        return self._loss

    @property
    def y_hat(self):

        if self._y_hat is None:
            self.symbolic_loss()

        return self._y_hat

    @property
    def weights(self):

        if len(self._weights) == 0:
            self.symbolic_loss()

        return self._weights

    @property
    def biases(self):

        if len(self._biases) == 0:
            self.symbolic_loss()

        return self._biases