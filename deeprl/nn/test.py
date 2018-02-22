
"""
-- Unit/regression tests for currently supported architectures
"""

import sys
import time
import logging as log
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as data

INTERACTIVE_MODE = True
logging_name = "nn-test"

if INTERACTIVE_MODE:
    sys.path.extend(['./deeprl/nn/'])
    logging_loc = "./logs/"
else:
    logging_loc = "../../logs/"

logging_loc += "-".join(time.asctime().split(" ")) + logging_name + ".log"

from nn import NN

# set up logging, set new excepthook
l = log.getLogger(logging_name)
file_handle = log.FileHandler(logging_loc)
file_handle.setFormatter(log.Formatter("%(asctime)s: %(message)s"))
l.addHandler(file_handle)

sys.excepthook = lambda *args: log.getLogger(logging_name)\
    .error("Exception caught: ", exec_info=args)

# load mnist data
mnist_data = data.read_data_sets('MNIST_data', one_hot=True)


##################
# base test class

class NNTest(object):

    def __init__(self, config):

        self._config = config

    def run(self, logger=None):

        graph_ = NN(self._config)
        training_step = graph_.optimize(graph_.loss, "rmsprop", 0.01)

        # start session, train with mnist training data
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        final_loss = None
        for i in range(1000):
            batch = mnist_data.train.next_batch(100)
            _, loss_val = sess.run(
                    [training_step, graph_.loss],feed_dict={
                        graph_._x_placeholder: batch[0],
                        graph_._y_placeholder: batch[1]
                    })
            final_loss = loss_val

            if i % 100 == 0:
                print "loss: " + str(loss_val)

        return final_loss


##########
# TEST 1 #
##########

# standard feedforward network/mlp
# two hidden layers with relu activation, cross entropy loss

mlp_config = {
    "layers": [
        {
            "type": "input",
            "dim": 784
        },
        {
            "type": "dense",
            "activation": "relu",
            "dim": 100
        },
        {
            "type": "dense",
            "activation": "relu",
            "dim": 30
        },
        {
            "type": "output",
            "activation": "linear",
            "dim": 10
        }
    ],
    "options": {
        "loss_type": "cross_entropy"
    }
}

mlp = NNTest(mlp_config)
loss_ = mlp.run()

if loss_ is not None:
    l.info("MLP - PASS")


##########
# TEST 2 #
##########

# feedforward convolutional network
# one conv layer with relu activation, one pooling, cross-entropy loss

conv_config = {
    "layers": [
        {
            "type": "input",
            "dim": 784
        },
        {
            "type": "conv",
            "pixel_dim": [28,28],
            "strides": 5,
            "channels": [1, 32, 64],
            "stack_depth": 2,
            "pooling": "true",
            "pool_dim": [[2,2], [2,2]]
        },
        {
            "type": "dense",
            "activation": "relu",
            "dim": 1024
        },
        {
            "type": "output",
            "activation": "linear",
            "dim": 10
        }
    ],
    "options": {
        "loss_type": "cross_entropy"
    }
}

conv_net = NNTest(conv_config)
loss_ = conv_net.run()

if loss_ is not None:
    l.info("CONV - PASS")