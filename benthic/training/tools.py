r"""A collection of tools designed for training module."""

import yaml

from itertools import product
from pathlib import Path
from typing import Any, Dict, List


def load_configuration(path: Path) -> List[Dict[str, Any]]:
    r"""Load all combinations of parameters from a YAML configuration file."""

    def generate_combinations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
        r"""Recursively generate parameter combinations."""
        if isinstance(d, dict):
            combinations = {k: generate_combinations(v) for k, v in d.items()}
            keys, values = zip(*combinations.items())
            return [dict(zip(keys, combo)) for combo in product(*values)]
        return d if isinstance(d, list) else [d]

    # Open and read the YAML configuration file
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    # Generate combinations
    return generate_combinations(config)


def prepare_wandb_config(config: dict) -> dict:
    r"""Transform nested dictionaries in simple dictionary

    Some parameters are nested in the config dictionary,
    this function flattens them and adds some more explicit
    information for better readability in WandB.

    Arguments:
        config: Training configuration dictionary.

    Returns:
        Dictionary for wandb.init(config=...) parameter.
    """
    model_type = config["config_model"]["model_type"]

    # Determine model configuration based on type
    if model_type == "gp":
        model_label = "Gaussian Process"
        model_config = config["config_gp"]
    else:
        model_label = "Neural Network"
        model_config = {
            "epochs": config["config_nn"]["epochs"],
            "learning_rate": config["config_nn"]["learning_rate"],
            "nb_layers": len(config["config_nn"]["layers"]),
            "dropout": config["config_nn"]["dropout"],
            "num_components": config["config_nn"]["num_components"],
            "num_inference_samples": config["config_nn"]["num_inference_samples"],
            **{
                f"layer_{i + 1}_width": neurons
                for i, neurons in enumerate(config["config_nn"]["layers"])
            },
        }

    return {
        "Dataset": config["config_dataset"],
        "Model": model_label,
        "Configuration": model_config,
    }
