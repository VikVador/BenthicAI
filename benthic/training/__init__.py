r"""A tool for training models."""

import wandb
import xarray as xr

from benthic.config import PATH_DATASET_TRAIN_VAL
from benthic.data.const import DATASET_INPUT_VARIABLES, DATASET_OUTPUT_VARIABLES
from benthic.data.dataset import get_datasets
from benthic.data.tools import filter_outliers, from_zarr_to_numpy
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

    # Extract preprocessing parameters (not forwarded to get_datasets)
    cfg_ds = config["config_dataset"]
    stats = cfg_ds.get("stats", None)
    period = cfg_ds.get("period", "annual")
    q_min = cfg_ds.get("q_min", 0.0)
    q_max = cfg_ds.get("q_max", 100.0)
    zero_safe = cfg_ds.get("zero_safe", True)

    # Extracting the input and output variables
    X, y = (
        from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES, stats=stats, period=period),
        from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES),
    )

    # Filtering outliers on inputs only (outputs are trait values in [0, 1])
    X = filter_outliers(X, q_min=q_min, q_max=q_max, zero_safe=zero_safe)

    # Create nested k-fold splits (structural params only)
    _DATASET_KEYS = ("r_validation", "k_validation", "k_test", "seed_validation", "seed_test")
    dataset_kwargs = {k: cfg_ds[k] for k in _DATASET_KEYS}
    X_train, y_train, X_val, y_val, _, _, _, coords_val, _ = get_datasets(X, y, **dataset_kwargs)

    # Extracting the current fold for training and validation
    X_train, y_train, X_val, y_val, coords_val = (
        X_train[fold_idx],
        y_train[fold_idx],
        X_val[fold_idx],
        y_val[fold_idx],
        coords_val[fold_idx],
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
            coords_val=coords_val,
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
            coords_val=coords_val,
            model_folder=model_folder,
            fold_idx=fold_idx,
            config=config,
        )

    else:
        raise ValueError(f"Unknown model type: {config['config_model']}")

    # Closing WandB run
    wandb.finish()
