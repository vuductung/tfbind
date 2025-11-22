import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_config):
    """Get device based on config."""
    if device_config == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device_config


def get_paths(environment):
    """Get paths based on environment (local or raven)"""

    config_file = Path(__file__).parent.parent.parent / "configs" / "path.yml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
        paths = config[environment]

    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return paths
