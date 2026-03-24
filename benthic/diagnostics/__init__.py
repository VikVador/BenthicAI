r"""A tool for evaluating models."""

import numpy as np

from benthic.diagnostics.metrics import bias, r2_score, rmse


def diagnostics(fold_idx: int, model_folder: str) -> None:
    r"""Compute diagnostics for a given fold and model.

    Arguments:
        fold_idx: Index of the fold for k-fold cross-validation (inner loop).
        model_folder: Path to the folder where the model results are saved.
    """

    # Loading the data
    y_true = np.load(model_folder / f"{fold_idx}_y_true.npy")
    y_pred_mean = np.load(model_folder / f"{fold_idx}_y_pred_mean.npy")

    y_standardized_true = np.load(model_folder / f"{fold_idx}_standardized_y_true.npy")
    y_standardized_pred_mean = np.load(model_folder / f"{fold_idx}_standardized_y_pred_mean.npy")

    # Computing diagnostics
    score_r2 = r2_score(y_true, y_pred_mean)
    score_bias = bias(y_true, y_pred_mean)
    score_rmse = rmse(y_true, y_pred_mean)

    score_standardized_bias = bias(y_standardized_true, y_standardized_pred_mean)
    score_standardized_rmse = rmse(y_standardized_true, y_standardized_pred_mean)

    # Saving diagnostics per fold, model and sample
    np.save(model_folder / f"{fold_idx}_bias.npy", score_bias)
    np.save(model_folder / f"{fold_idx}_r2.npy", score_r2)
    np.save(model_folder / f"{fold_idx}_rmse.npy", score_rmse)

    # Saving diagnostics per fold, model and sample
    np.save(model_folder / f"{fold_idx}_standardized_bias.npy", score_standardized_bias)
    np.save(model_folder / f"{fold_idx}_standardized_rmse.npy", score_standardized_rmse)
