r"""A collection of tools to train a Gaussian Process."""

import numpy as np

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


def kernel_from_string(expr: str):
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


def nlpd_per_output(y_true, y_pred, y_std):
    y_std = y_std[:, np.newaxis]  # (n_test, 1) for correct broadcasting
    pointwise_nlpd = 0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * ((y_true - y_pred) ** 2) / (
        y_std**2
    )
    return np.mean(pointwise_nlpd, axis=0)  # (n_outputs,)


def training_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_folder: str,
    fold_idx: int,
    config: dict,
):
    r"""Train a Gaussian Process.

    Arguments:
        X_train: Input data for training.
        y_train: Output data for training.
        X_val: Input data for validation.
        y_val: Output data for validation.
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

    # Initializing of Gaussian Process Regressor
    fold_kernel = kernel_from_string(config["config_gp"]["kernel"])
    fold_gpr = GaussianProcessRegressor(
        kernel=fold_kernel,
        alpha=float(config["config_gp"]["alpha"]),
        n_restarts_optimizer=int(config["config_gp"]["n_restarts_optimizer"]),
    )

    # Training
    fold_gpr.fit(X_train, y_train)

    # Predictions on train and validation
    y_pred_tr, y_std_tr = fold_gpr.predict(X_train, return_std=True)
    y_pred_va, y_std_va = fold_gpr.predict(X_val, return_std=True)

    # Saving results and validation set
    pred_path = model_folder / f"gp_fold_{fold_idx:03d}_predictions.npz"
    np.savez(
        pred_path,
        y_val=y_val,
        y_pred_va=y_pred_va,
        y_std_va=y_std_va,
    )
