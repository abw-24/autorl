
"""
-- Various utilities for easily specifying and applying different activation
 functions and optimization algorithms via string arguments
"""

import tensorflow as tf
import sys


def optimize(loss, optimizer_type, learning_rate, momentum=0.2):
    """

    :param loss:
    :param optimizer:
    :param momentum:
    :return:
    """

    optimizers_ = {"sgd": tf.train.GradientDescentOptimizer,
                   "adam": tf.train.AdamOptimizer,
                   "adagrad": tf.train.AdagradOptimizer,
                   "adadelta": tf.train.AdadeltaOptimizer,
                   "rmsprop": tf.train.RMSPropOptimizer,
                   "momentum": tf.train.MomentumOptimizer}
    try:
        optim = optimizers_[optimizer_type]
    except KeyError:
        print 'Unrecognized optimizer specification. Please check configuration.'
        sys.exit()

    if optimizer_type == 'momentum':
        return optim(learning_rate=learning_rate, momentum=momentum).minimize(loss)
    else:
        return optim(learning_rate=learning_rate).minimize(loss)


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


def build_placeholders(x_dim, batch_size):
    """

    :param x_dim:
    :param batch_size:
    :return:
    """

    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, x_dim))

    y_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return x_placeholder, y_placeholder