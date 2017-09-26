
"""
-- Class tfUtilities houses base static methods -
-- Inherited by all graph constructor classes
"""

import tensorflow as tf
import cPickle as pkl
import sys


class tfUtilities(object):

    @staticmethod
    def build_placeholders(x_dim, batch_size):
        """

        :param x_dim:
        :param batch_size:
        :return:
        """

        x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, x_dim))

        y_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

        return x_placeholder, y_placeholder

    @staticmethod
    def activate(op, activation_type):
        """

        :param op:
        :param activation_type:
        :return:
        """

        activation_functions_ = {"tanh": tf.tanh,
                                "relu": tf.nn.relu,
                                "sigmoid": tf.sigmoid}

        try:
            activation = activation_functions_[activation_type]
        except KeyError:
            print 'Unrecognized activation specification. Please check configuration.'
            sys.exit()

        return activation(op)

    @staticmethod
    def compute_loss(output, labels, loss_type):
        """

        :param output:
        :param labels:
        :param loss_type:
        :return:
        """

        square_loss = lambda x,y: tf.square(y - x)

        loss_functions = {'cross_entropy': tf.nn.softmax_cross_entropy_with_logits,
                          'squared_loss': square_loss}

        return loss_functions[loss_type](output, labels)

    @staticmethod
    def optimize(loss, optimizer_type, learning_rate, momentum=0.2):
        """

        :param loss:
        :param optimizer:
        :param momentum:
        :return:
        """

        optimizers_ = {
            "sgd": tf.train.GradientDescentOptimizer,
            "adam": tf.train.AdamOptimizer,
            "adagrad": tf.train.AdagradOptimizer,
            "adadelta": tf.train.AdadeltaOptimizer,
            "rmsprop": tf.train.RMSPropOptimizer,
            "momentum": tf.train.MomentumOptimizer

        }

        try:
            optim = optimizers_[optimizer_type]
        except KeyError:
            print 'Unrecognized optimizer specification. Please check configuration.'
            sys.exit()

        if optimizer_type == 'momentum':
            return optim(learning_rate=learning_rate, momentum=momentum).minimize(loss)
        else:
            return optim(learning_rate=learning_rate).minimize(loss)

    @staticmethod
    def model_save(path, batch, out_weights, out_biases):
        """

        :param path:
        :param epoch:
        :param out_weights:
        :param out_biases:
        :param loss:
        :return:
        """

        if path[-1] != '/':
            path += '/'

        with open(path + "batch_" + str(batch) + "_weights.pkl", "wb") as weight_f:
            pkl.dump(out_weights, weight_f)

        with open(path + "batch_" + str(batch) + "_biases.pkl", "wb") as bias_f:
            pkl.dump(out_biases, bias_f)