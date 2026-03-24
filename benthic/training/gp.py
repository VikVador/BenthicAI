r"""A collection of tools to train a Gaussian Process."""

import numpy as np
import wandb

from pathlib import Path
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
from sklearn.metrics import r2_score


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


def smape_per_output(y_true, y_pred, eps=1e-8):
    # Symmetric MAPE in %, robust when targets can be near zero
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    smape = 200.0 * np.abs(y_pred - y_true) / denom
    return np.mean(smape, axis=0)  # (n_outputs,)


def training_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    model_folder: str,
    config: dict,
    fold_idx: int,
):
    """Train a Gaussian Process.

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

    fold_kernel = kernel_from_string(config["config_gp"]["kernel"])
    fold_gpr = GaussianProcessRegressor(
        kernel=fold_kernel,
        alpha=float(config["config_gp"]["alpha"]),
        n_restarts_optimizer=int(config["config_gp"]["n_restarts_optimizer"]),
    )
    fold_gpr.fit(X_train, y_train)

    # Predictions on train and validation
    y_pred_tr, y_std_tr = fold_gpr.predict(X_train, return_std=True)
    y_pred_va, y_std_va = fold_gpr.predict(X_val, return_std=True)

    # Saving validation predictions and true values (sample x predictor)
    model_folder_path = Path(model_folder)
    model_folder_path.mkdir(parents=True, exist_ok=True)
    code = str(fold_idx)

    y_true_to_save = np.asarray(y_val)
    y_pred_to_save = np.asarray(y_pred_va)
    y_var_to_save = np.asarray(y_std_va) ** 2

    if y_true_to_save.ndim == 1:
        y_true_to_save = y_true_to_save[:, np.newaxis]
    if y_pred_to_save.ndim == 1:
        y_pred_to_save = y_pred_to_save[:, np.newaxis]
    if y_var_to_save.ndim == 1:
        y_var_to_save = y_var_to_save[:, np.newaxis]

    y_true_path = model_folder_path / f"{code}_y_true.npy"
    y_pred_path = model_folder_path / f"{code}_y_pred.npy"
    y_var_path = model_folder_path / f"{code}_y_var.npy"

    np.save(y_true_path, y_true_to_save)
    np.save(y_pred_path, y_pred_to_save)
    np.save(y_var_path, y_var_to_save)

    print(f"|- Saved validation targets to: {y_true_path}")
    print(f"|- Saved validation predictions to: {y_pred_path}")
    print(f"|- Saved validation variance to: {y_var_path}")

    # Per-output metrics (train)
    rmse_per_output_tr = np.sqrt(np.mean((y_train - y_pred_tr) ** 2, axis=0))
    nlpd_per_output_tr = nlpd_per_output(y_train, y_pred_tr, y_std_tr)
    smape_per_output_tr = smape_per_output(y_train, y_pred_tr)
    r2_per_output_tr = r2_score(y_train, y_pred_tr, multioutput="raw_values")

    # Per-output metrics (validation)
    rmse_per_output_va = np.sqrt(np.mean((y_val - y_pred_va) ** 2, axis=0))
    nlpd_per_output_va = nlpd_per_output(y_val, y_pred_va, y_std_va)
    smape_per_output_va = smape_per_output(y_val, y_pred_va)
    r2_per_output_va = r2_score(y_val, y_pred_va, multioutput="raw_values")

    # Global metrics (train)
    rmse_global_tr = float(np.sqrt(np.mean((y_train - y_pred_tr) ** 2)))
    nlpd_global_tr = float(np.mean(nlpd_per_output_tr))
    smape_global_tr = float(np.mean(smape_per_output_tr))
    r2_global_tr = float(np.mean(r2_per_output_tr))

    # Global metrics (validation)
    rmse_global_va = float(np.sqrt(np.mean((y_val - y_pred_va) ** 2)))
    nlpd_global_va = float(np.mean(nlpd_per_output_va))
    smape_global_va = float(np.mean(smape_per_output_va))
    r2_global_va = float(np.mean(r2_per_output_va))

    # W&B logging: global + per-output metrics
    if config.get("use_wandb", True) and wandb.run is not None:

        def _sanitize_metric_name(name: str) -> str:
            return (
                str(name)
                .replace(" ", "_")
                .replace("/", "_")
                .replace("%", "pct")
                .replace(".", "_")
                .replace("-", "_")
            )

        metrics = {
            "global/train_rmse": rmse_global_tr,
            "global/train_nlpd": nlpd_global_tr,
            "global/train_smape": smape_global_tr,
            "global/train_r2": r2_global_tr,
            "global/val_rmse": rmse_global_va,
            "global/val_nlpd": nlpd_global_va,
            "global/val_smape": smape_global_va,
            "global/val_r2": r2_global_va,
        }

        output_names = config.get("output_names", [])
        for i, name in enumerate(output_names):
            if i >= len(rmse_per_output_tr):
                break
            n = _sanitize_metric_name(name)
            metrics[f"per_output/train/rmse/{n}"] = float(rmse_per_output_tr[i])
            metrics[f"per_output/train/nlpd/{n}"] = float(nlpd_per_output_tr[i])
            metrics[f"per_output/train/smape/{n}"] = float(smape_per_output_tr[i])
            metrics[f"per_output/train/r2/{n}"] = float(r2_per_output_tr[i])

            metrics[f"per_output/val/rmse/{n}"] = float(rmse_per_output_va[i])
            metrics[f"per_output/val/nlpd/{n}"] = float(nlpd_per_output_va[i])
            metrics[f"per_output/val/smape/{n}"] = float(smape_per_output_va[i])
            metrics[f"per_output/val/r2/{n}"] = float(r2_per_output_va[i])

        wandb.log(metrics)
