r"""A collection of tools to train a Gaussian Process."""

import numpy as np


def training_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    model_folder: str,
    config: dict,
):
    r"""Train a Gaussian Process.

    Arguments:
        X_train: Input data for training.
        y_train: Output data for training.
        X_val: Input data for validation.
        y_val: Output data for validation.
        model_name: Name of the model to be trained, used for logging and saving.
        model_folder: Path to the folder where the model will be saved.
        config: Training configuration dictionary.
    """

    # Displaying information over terminal (1)
    print(f"Training model: {model_name}")
    print(f"|- Model folder: {model_folder}")
    print(f"|- Model type: {config['config_model']}")
    print(f"|- Training dataset shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"|- Validation dataset shape: X_val={X_val.shape}, y_val={y_val.shape}")
    print("|- Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
