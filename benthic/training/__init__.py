r"""A collection of tools to train a model."""

import wandb
import xarray as xr

from benthic.config import FOLDER_MODEL_GP, FOLDER_MODEL_NN, PATH_DATASET_TRAIN_VAL
from benthic.data.const import DATASET_INPUT_VARIABLES, DATASET_OUTPUT_VARIABLES
from benthic.data.dataset import get_datasets
from benthic.data.tools import Statistics, from_zarr_to_numpy
from benthic.training.gp import training_gp
from benthic.training.nn import training_nn


def training(model_name: str, fold_idx: int, config: dict):
    r"""Function to train a model.

    Arguments:
        model_name: Name of the model to be trained, used for logging and saving.
        fold_idx: Index of the fold for k-fold cross-validation (inner loop).
        config: Training configuration dictionary.
    """

    # ===================
    #   INITIALIZATION
    # ===================
    # Initialization of WandB
    wandb.init(
        mode=config["config_wandb"]["mode"],
        project=config["config_wandb"]["project"],
        entity=config["config_wandb"]["team"],
        group=model_name,
        name=f"{model_name}",
        config=config,
    )

    # Creation of model folder to save model and results
    model_folder = (
        FOLDER_MODEL_NN / f"{model_name}"
        if config["config_model"] == "nn"
        else FOLDER_MODEL_GP / f"{model_name}"
    )

    # ===================
    #       DATASET
    # ===================
    # Displaying information over terminal (1)
    print(f"MODEL : {model_name}")
    print(f"|- Folder Index : {fold_idx}")
    print("|- Loading and standardizing dataset...")

    # Loading the dataset
    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)

    # Extracting the input and output variables
    X, y = (
        from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES),
        from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES),
    )

    # Standardizing data
    stats_X, stats_y = Statistics(), Statistics()
    stats_X.fit(X), stats_y.fit(y)
    X_standardized, y_standardized = stats_X.standardize(X), stats_y.standardize(y)

    # Create nested k-fold splits using standardized data
    X_train, y_train, X_val, y_val, _, _ = get_datasets(
        X_standardized,
        y_standardized,
        **config["config_dataset"],
    )

    # Extracting the fold index for the current training job
    X_train, y_train, X_val, y_val = (
        X_train[fold_idx],
        y_train[fold_idx],
        X_val[fold_idx],
        y_val[fold_idx],
    )

    # Displaying information over terminal (2)
    print("|- Dataset loaded and standardized...")
    print("|- Launching training...")

    # ===================
    #      TRAINING
    # ===================
    # Launching training based on the model type specified in the configuration
    if config["config_model"]["model_type"] == "nn":
        training_nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            model_folder=model_folder,
            config=config,
        )

    elif config["config_model"]["model_type"] == "gp":
        training_gp(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
            model_folder=model_folder,
            config=config,
        )

    else:
        raise ValueError(f"Unknown model type: {config['config_model']}")

    # Closing WandB run
    wandb.finish()
