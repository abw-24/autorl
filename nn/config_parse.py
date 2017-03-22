
"""
-- Layer configuration parser
"""

import json
from copy import deepcopy


def read_config(config_loc):

    with open(config_loc, "rb") as f:
        raw_config = json.load(f)

    return raw_config


class layerParse(object):

    def __init__(self, model_type, layer_dict):

        self._model_type = model_type
        self._raw_layers = layer_dict

    def _parse(self):

        out = []
        for l in self._raw_layers:
            tmp_layer = deepcopy(l)
            









