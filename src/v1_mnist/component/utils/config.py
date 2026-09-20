import yaml
import os
import random
import numpy as np
import torch
from types import SimpleNamespace

def merge_dicts(base, override):
    """Merge weekly settings on top of base settings."""
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path):

    # --------------------------------------------------
    # Load the requested weekly config
    # --------------------------------------------------
    with open(config_path, 'r') as f:
        weekly_config = yaml.safe_load(f) or {}

    # --------------------------------------------------
    # Automatically load base.yaml from same folder
    # --------------------------------------------------
    config_dir = os.path.dirname(os.path.abspath(config_path))
    base_path = os.path.join(config_dir, 'base.yaml')

    if os.path.basename(config_path) != 'base.yaml' and os.path.exists(base_path):
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f) or {}

        config_dict = merge_dicts(base_config, weekly_config)
    else:
        config_dict = weekly_config

    # --------------------------------------------------
    # Automatically set output folders by week
    # --------------------------------------------------
    if 'run' in config_dict and 'name' in config_dict['run']:

        week = config_dict['run']['name']

        output_root = config_dict.get(
            'project', {}
        ).get(
            'output_root',
            './outputs'
        )

        if 'paths' not in config_dict:
            config_dict['paths'] = {}

        output_dir = os.path.join(output_root, week)

        config_dict['paths'].setdefault('output_dir', output_dir)
        config_dict['paths'].setdefault('checkpoints_dir', os.path.join(output_dir, 'checkpoints'))
        config_dict['paths'].setdefault('embeddings_dir', os.path.join(output_dir, 'embeddings'))
        config_dict['paths'].setdefault('figures_dir', os.path.join(output_dir, 'figures'))
        config_dict['paths'].setdefault('tables_dir', os.path.join(output_dir, 'tables'))
        config_dict['paths'].setdefault('som_models_dir', os.path.join(output_dir, 'som_models'))
        config_dict['paths'].setdefault('logs_dir', os.path.join(output_dir, 'logs'))
        config_dict['paths'].setdefault('analysis_dir', os.path.join(output_dir, 'analysis'))

    # --------------------------------------------------
    # Convert dict to nested SimpleNamespace
    # --------------------------------------------------
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
