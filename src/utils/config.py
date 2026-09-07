import yaml
import os
import random
import numpy as np
import torch
from types import SimpleNamespace

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to nested SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = dict_to_namespace(v)
            return SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_namespace(i) for i in d]
        else:
            return d
            
    return dict_to_namespace(config_dict)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(config):
    if hasattr(config, 'device') and config.device != 'auto':
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
