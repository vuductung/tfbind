import os
from pathlib import Path

import yaml


def get_paths(environment):
    """Get paths based on environment (local or raven)"""

    config_file = Path(__file__).parent.parent / "configs" / "paths.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
        paths = config[environment]

    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return paths
