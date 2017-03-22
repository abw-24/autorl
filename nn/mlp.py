

"""
-- Constructor class for computing symbolic loss for a given MLP.
Takes a JSON configuration to specify layers and options with
the following conventions:

1. Top level keys:
-- "layers" - an array containing layer by layer key/value specifications
(in "descending" order) for constructing the mlp graph. The following
layer types are available: "input", "hidden", "output"

"""

import tensorflow as tf
from tf_utilities import tfUtilities


class MLP(tfUtilities):

    def __init__(self, x_placeholder, y_placeholder, config_map):

        self._x = x_placeholder
        self._y = y_placeholder
        self._layers = config_map['layers']
        self._loss_type = config_map['loss']
        self._loss = None
        self._parameters = []