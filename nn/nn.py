

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

# todo: add conv/pool layers
# todo: add autoencoder layers
# todo: add training


class LayerCompute(tfUtilities):

    def __init__(self, layer_configuration, in_weights=None):

        self._layer_configs = layer_configuration
        self._in_weights = in_weights

        assert self._layer_configs[0]["type"] == "input", "First layer must be of type 'input'"
        assert self._layer_configs[-1]["type"] == "output", "Last layer must be of type 'output'"

        self._input_dim = self._layer_configs[0]["dim"]
        self._output_nodes = self._layer_configs[-1]["dim"]

        # now just take hidden layers
        self._layer_configs = self._layer_configs[1:]

        # if we have input weights, should match up after removing input layer configuration
        if self._in_weights is not None:
            assert len(self._in_weights) == len(self._layer_configs), "Input weights must have length n_layers - 1"

        self._x_placeholder, self._y_placeholder = self.build_placeholders(self._input_dim)

        self._layer_compute_ops = {
            "input": self._input_layer,
            "output": self._output_layer,
            "dense": self._dense_hidden_layer,
            "conv": self._convolution_layer
        }

        self._layer_weights = []

        self._previous_dim = None
        self._i = 0

    def _input_layer(self):

        assert self._i == 0, "Input layer specified incorrectly. Exiting."

        self._previous_dim = self._input_dim
        self._i += 1

        return self._x_placeholder

    def _dense_hidden_layer(self, in_graph, layer_config, in_w=None):

        _col_dim = layer_config["dim"]
        _act = layer_config["activation"]

        with tf.name_scope("dense_" + str(self._i) + "_"):

            if in_w is None:
                # using truncated normals for weight initialization, zeros for biases
                w_dim = [self._previous_dim, _col_dim]
                sd = 1.0 / math.sqrt(float(self._previous_dim))
                weights = tf.Variable(tf.truncated_normal(w_dim, stddev=sd, name='weights'))
                biases = tf.Variable(tf.zeros([_col_dim]), name='biases')

            else:
                weights = tf.Variable(in_w["weights"].astype(np.float32), name='weights')
                biases = tf.Variable(in_w["biases"].astype(np.float32), name='biases')

            self._layer_weights.append({"weights": weights, "biases": biases})

            linear_op = tf.matmul(in_graph, weights) + biases
            nonlinear_op = self.activate(linear_op, _act)

        self._i += 1
        self._previous_dim = _col_dim

        return nonlinear_op

    def _convolution_layer(self, in_graph, layer_config, in_w=None):
        raise NotImplementedError('Conv nets not yet available')

    def _pooling_layer(self, in_graph, layer_config, in_w=None):
        raise NotImplementedError('Conv nets not yet available')

    def _output_layer(self, in_graph, layer_config, in_w=None):

        _output_dim = layer_config["dim"]

        with tf.name_scope("output_" + str(self._i) + "_"):

            if in_w is None:
                # using truncated normals for weight initialization, zeros for biases
                w_dim = [self._previous_dim, _output_dim]
                sd = 1.0 / math.sqrt(float(self._previous_dim))
                weights = tf.Variable(tf.truncated_normal(w_dim, stddev=sd, name='weights'))
                biases = tf.Variable(tf.zeros([_output_dim]), name='biases')

            else:
                weights = tf.Variable(in_w["weights"].astype(np.float32), name='weights')
                biases = tf.Variable(in_w["biases"].astype(np.float32), name='biases')

            self._layer_weights.append({"weights": weights, "biases": biases})

            linear_op = tf.matmul(in_graph, weights) + biases

        self._i += 1
        self._previous_dim = _output_dim

        return linear_op

    def _layer_compute(self, in_graph, layer_config, in_w=None):

        _type = layer_config['type']
        _tmp_layer_op = self._layer_compute_ops[_type]

        # apply op to graph
        return _tmp_layer_op(in_graph, layer_config, in_w)

    def _model_compute(self):

        # run the input layer
        _out_graph = self._input_layer()

        for i, c in enumerate(self._layer_configs):

            if self._in_weights is not None:
                _out_graph = self._layer_compute(_out_graph, c, self._in_weights[i])

            else:
                _out_graph = self._layer_compute(_out_graph, c)

        return _out_graph


class NN(LayerCompute):

    def __init__(self, model_ops, in_weights=None, wait=False):
        """
        Builds the computatuional graph for a multi-layer perception

        :param wait:
        :param in_weights:
        :param model_ops:
        :return:
        """

        super(NN, self).__init__(model_ops['layers'], in_weights)

        self._ops = model_ops['options']

        # parse global architecture options
        self._loss_type = self._ops['loss_type']

        # set up instance objects to be filled with later
        self._loss = None
        self._y_hat = None

        if not wait:
            self.build()

    def build(self):
        """

        :return:
        """

        self._y_hat = self._model_compute()

        batch_loss = self.compute_loss(self._y_hat, self._y_placeholder, self._loss_type)
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

        if len(self._layer_weights) == 0:
            self.build()

        return self._layer_weights
