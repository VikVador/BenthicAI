r"""A collection of tools to train a model."""

import wandb
import xarray as xr

from benthic.config import PATH_DATASET_TRAIN_VAL
from benthic.data.const import DATASET_INPUT_VARIABLES, DATASET_OUTPUT_VARIABLES
from benthic.data.dataset import get_datasets
from benthic.data.tools import from_zarr_to_numpy
from benthic.training.gp import training_gp
from benthic.training.nn import training_nn
from benthic.training.tools import prepare_wandb_config


def training(fold_idx: int, model_name: str, model_folder: str, config: dict):
    r"""Launch the training of a model.

    Arguments:
        fold_idx: Index of the fold for k-fold cross-validation (inner loop).
        model_name: Name of the model to be trained, used for openning a WandB run.
        model_folder: Path to the modelfolder where everything is saved.
        config: Training configuration dictionary.
    """

    wandb.init(
        mode=config["config_wandb"]["mode"],
        project=config["config_wandb"]["project"],
        entity=config["config_wandb"]["team"],
        group=model_name,
        name=f"{model_name}",
        config=prepare_wandb_config(config),
    )

    # ===================
    #       DATASET
    # ===================
    # Loading the dataset
    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)

    # Extracting the input and output variables
    X, y = (
        from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES),
        from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES),
    )

    # Create nested k-fold splits
    X_train, y_train, X_val, y_val, _, _ = get_datasets(X, y, **config["config_dataset"])

    # Extracting the current fold for training and validation
    X_train, y_train, X_val, y_val = (
        X_train[fold_idx],
        y_train[fold_idx],
        X_val[fold_idx],
        y_val[fold_idx],
    )

    # ===================
    #      TRAINING
    # ===================
    if config["config_model"]["model_type"] == "nn":
        training_nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_folder=model_folder,
            fold_idx=fold_idx,
            config=config,
        )

    elif config["config_model"]["model_type"] == "gp":
        training_gp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_folder=model_folder,
            fold_idx=fold_idx,
            config=config,
        )

    else:
        raise ValueError(f"Unknown model type: {config['config_model']}")

    # Closing WandB run
    wandb.finish()
