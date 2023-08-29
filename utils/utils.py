import random
import numpy as np
import torch

import json

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data




def flatten_dict(d):
    if type(d)==dict:   
        res = [[i]+flatten_dict(d[i]) for i in d]
        return [j for i in res for j in i]
    return [str(d)]

def flatten_list(l):
    res = [flatten_list(i) if type(i)==list else [i] for i in l]
    return [j for i in res for j in i]


class AttrDict:
    def __init__(self, data_dict):
        self.__dict__['_data_dict'] = data_dict

    def __getattr__(self, key):
        if key in self._data_dict:
            return self._data_dict[key]
        raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == '_data_dict':
            # Ensure the '_data_dict' attribute is set using the standard mechanism
            super().__setattr__(key, value)
        else:
            self._data_dict[key] = value

    def __iter__(self):
        return iter(self._data_dict)