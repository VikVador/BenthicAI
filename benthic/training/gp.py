r"""A collection of tools to train a Gaussian Process."""

import numpy as np
import wandb

from benthic.data.tools import Statistics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)


# fmt: off
def kernel_from_string(expr: str) -> object:
    r"""Parse a kernel expression from a string.

    Arguments:
        expr: A string expression representing the kernel, e.g., "ConstantKernel(1.0)"
    """

    allowed = {
        "RBF": RBF,
        "Matern": Matern,
        "WhiteKernel": WhiteKernel,
        "ConstantKernel": ConstantKernel,
        "RationalQuadratic": RationalQuadratic,
        "ExpSineSquared": ExpSineSquared,
        "DotProduct": DotProduct,
    }

    # Only use this for trusted config files
    return eval(expr, {"__builtins__": {}}, allowed)


def training_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    coords_val: np.ndarray,
    model_folder: str,
    fold_idx: int,
    config: dict,
):
    """Train a Gaussian Process.

    Arguments:
        X_train: Input data for training.
        y_train: Output data for training.
        X_val: Input data for validation.
        y_val: Output data for validation.
        coords_val: Coordinates (lat, lon, depth) of the validation stations, shape (n_val, 3).
        model_folder: Path to the folder where the model will be saved.
        fold_idx: Index of the fold for k-fold cross-validation (inner loop).
        config: Training configuration dictionary.
    """

    # Standardizing data (using training set statistics to leak no information from validation set)
    stats_X, stats_y = Statistics(), Statistics()
    stats_X.fit(X_train), stats_y.fit(y_train)
    X_train, y_train, X_val, y_val = (
        stats_X.standardize(X_train),
        stats_y.standardize(y_train),
        stats_X.standardize(X_val),
        stats_y.standardize(y_val),
    )

    # Initializing the Gaussian Process Regressor
    fold_kernel = kernel_from_string(config["config_gp"]["kernel"])
    fold_gpr = GaussianProcessRegressor(
        kernel=fold_kernel,
        alpha=float(config["config_gp"]["alpha"]),
        n_restarts_optimizer=int(config["config_gp"]["n_restarts_optimizer"]),
    )

    # Training
    fold_gpr.fit(X_train, y_train)

    # Predictions on train and validation
    y_mean_val, y_std_val = fold_gpr.predict(X_val, return_std=True)

    # Logging the final validation loss for the fold
    wandb.log({"Training & Validation / Final Validation Loss (GP)": y_mean_val.mean()})

    # Saving results and validation set (standardized)
    np.save(model_folder / f"{fold_idx}_standardized_y_true.npy",      y_val)
    np.save(model_folder / f"{fold_idx}_standardized_y_pred_mean.npy", y_mean_val)
    np.save(model_folder / f"{fold_idx}_standardized_y_pred_var.npy",  y_std_val ** 2)

    # Unscaling the data
    y_val, y_mean_val, y_std_val = (
        stats_y.unstandardize(y_val),
        stats_y.unstandardize(y_mean_val),
        stats_y.unstandardize(y_std_val),
    )

    # Saving results and validation set
    np.save(model_folder / f"{fold_idx}_y_true.npy",      y_val)
    np.save(model_folder / f"{fold_idx}_y_pred_mean.npy", y_mean_val)
    np.save(model_folder / f"{fold_idx}_y_pred_var.npy",  y_std_val ** 2)
    np.save(model_folder / f"{fold_idx}_coords.npy",      coords_val)
