

"""
-- Constructor class for computing symbolic loss for an arbitrary network.
Provides a caffe-like interface to Tensorflow in the sense that layers
 are JSON configurable

"""


import tensorflow as tf
from tf_utils import tfUtilities
import numpy as np

import math

#TODO: add handling for "depth" in conv layer


class FeedForward(tfUtilities):

    def __init__(self, layer_configuration, in_weights=None, in_graph=None):

        self._layer_configs = layer_configuration
        self._in_weights = in_weights
        self._in_graph = in_graph

        self._layer_compute_ops = {
            "input": self._input_layer,
            "output": self._dense_layer,
            "dense": self._dense_layer,
            "conv": self._convolution_layer
        }

        self._layer_weights = []

        self._previous_dim = None
        self._x_placeholder = None
        self._y_placeholder = None
        self._i = 0

    def _input_layer(self):

        input_dim = self._layer_configs[0]["dim"]

        self._x_placeholder, self._y_placeholder = self.build_placeholders(input_dim)

        # if we already have a graph, replace the x placeholder with it
        if self._in_graph:

            self._x_placeholder = self._in_graph

            if self._in_weights is not None:
                assert len(self._layer_configs) == len(self._in_weights), \
                    "If an input graph is specified and weights are provided, the number " \
                    "of layers and input weights should match in length."

        else:
            if self._in_weights is not None:
                assert len(self._layer_configs) == len(self._in_weights) - 1, \
                    "If weights are provided, the number of weight sets should be (#layers - 1)"

            self._layer_configs = self._layer_configs[1:]

        self._previous_dim = input_dim
        self._i += 1

        return self._x_placeholder

    def _dense_layer(self, in_graph, layer_config, in_w=None):
        """

        :param in_graph:
        :param layer_config:
        :param in_w:
        :return:
        """

        col_dim = int(layer_config["dim"])
        g = layer_config["activation"]

        with tf.name_scope("dense_" + str(self._i) + "_"):

            if in_w is None:
                # using truncated normals for weight initialization
                w_dim = [self._previous_dim, col_dim]
                sd = 1.0 / math.sqrt(float(self._previous_dim))
                weights = tf.Variable(tf.truncated_normal(w_dim, stddev=sd, mean=1.0, name='weights'))
                biases = tf.Variable(tf.truncated_normal([col_dim], stddev=sd, mean=1.0, name='biases'))

            else:
                weights = tf.Variable(in_w["weights"].astype(np.float32), name='weights')
                biases = tf.Variable(in_w["biases"].astype(np.float32), name='biases')

            self._layer_weights.append({"weights": weights, "biases": biases})

            linear_op = tf.matmul(in_graph, weights) + biases

            if g in ["none", "linear"]:
                out_op = linear_op
            else:
                out_op = self.activate(linear_op, g)

        self._i += 1
        self._previous_dim = col_dim

        return out_op

    @staticmethod
    def _pooling_op(in_graph, strides, padding="SAME"):
        """

        :param in_graph:
        :param layer_config:
        :param in_w:
        :return:
        """

        out_op = tf.nn.max_pool(in_graph,
                                ksize=[1, strides[0], strides[1], 1], strides=[1, strides[0], strides[1], 1],
                                padding=padding)
        return out_op

    @staticmethod
    def _conv_op(in_graph, weights, biases):

        conv_op = tf.nn.conv2d(in_graph, weights, strides=[1, 1, 1, 1], padding='SAME')
        bias_conv_op = tf.nn.bias_add(conv_op, biases)
        nonlinear_op = tf.nn.relu(bias_conv_op)

        return nonlinear_op

    def _convolution_layer(self, in_graph, layer_config, in_w=None):
        """

        :param in_graph:
        :param layer_config:
        :param in_w:
        :return:
        """

        # parse convolution layer options
        strides = layer_config['strides']
        pixel_dims = layer_config['pixel_dim']
        channels = layer_config['channels']
        p_bool = layer_config['pooling'].lower() in ["t", "true"]
        p_dim = layer_config['pool_dim']
        depth = int(layer_config["stack_depth"])

        try:
            padding = layer_config['padding']
        except KeyError:
            padding = "SAME"

        # set input reshape dim, check values for specification consistency, take
        # care of some handling for specifying stacked convolutions
        reshape_dim = [-1] + pixel_dims + [channels[0]]
        assert len(channels) == depth + 1, "Channels should be a list of lenth (depth + 1)"
        current_image_dim = pixel_dims

        if isinstance(strides, (str, int)):
            strides = [strides]*depth

        if not isinstance(p_dim[0], (list, tuple)):
            p_dim = [p_dim]*depth

        weight_set = []

        with tf.name_scope("stacked_conv_" + str(self._i) + "_"):

            stacked_op = tf.reshape(in_graph, shape=reshape_dim)

            for i, e in enumerate(zip(strides, p_dim)):

                s, p = e

                if in_w is None:
                    weights = tf.Variable(tf.truncated_normal([s, s, channels[i], channels[i+1]]), name="weights")
                    biases = tf.Variable(tf.truncated_normal([channels[i+1]]), name="biases")
                else:
                    weights = tf.Variable(in_w[i]["weights"].astype(np.float32), name='weights')
                    biases = tf.Variable(in_w[i]["biases"].astype(np.float32), name='biases')

                weight_set.append({"weights": weights, "biases": biases})

                stacked_op = self._conv_op(stacked_op, weights, biases)

                if p_bool:
                    stacked_op =  self._pooling_op(stacked_op, p, padding)
                    current_image_dim = [current_image_dim[0]/p[0], current_image_dim[1]/p[1]]

            out_dim = int(current_image_dim[0]*current_image_dim[1]*channels[-1])
            flattened_op = tf.reshape(stacked_op, [-1, out_dim])

            self._previous_dim = out_dim
            self._layer_weights.append(weight_set)

        self._i += 1

        return flattened_op

    def _layer_compute(self, in_graph, layer_config, in_w=None):
        """

        :param in_graph:
        :param layer_config:
        :param in_w:
        :return:
        """

        _type = layer_config['type']
        _tmp_layer_op = self._layer_compute_ops[_type]

        # apply op to graph
        return _tmp_layer_op(in_graph, layer_config, in_w)

    def _model_compute(self):
        """

        :return:
        """

        # run the input layer
        _out_graph = self._input_layer()

        for i, c in enumerate(self._layer_configs):

            if self._in_weights is not None:
                _out_graph = self._layer_compute(_out_graph, c, self._in_weights[i])

            else:
                _out_graph = self._layer_compute(_out_graph, c)

        return _out_graph


class NN(FeedForward):

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
