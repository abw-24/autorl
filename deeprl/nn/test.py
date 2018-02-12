
"""
-- Unit and regression tests for currently supported architectures
"""

INTERACTIVE_MODE = True

import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as data
import numpy as np
import math

if INTERACTIVE_MODE:
    sys.path.extend(['./deeprl/nn/'])

from nn import NN


# load mnist data
mnist_data = data.read_data_sets('MNIST_data', one_hot=True)


############################
# basic feedforward NN test

fully_connected_config = {
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
            "type": "output",
            "activation": "linear",
            "dim": 10
        }
    ],
    "options": {
        "loss_type": "cross_entropy"
    }
}

mlp_graph = NN(fully_connected_config)
training_step = mlp_graph.optimize(mlp_graph.loss, "rmsprop", 0.01)

# start session, train with mnist training data
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist_data.train.next_batch(100)
    training_step.run(feed_dict={
        mlp_graph._x_placeholder: batch[0],
        mlp_graph._y_placeholder: batch[1]
    })