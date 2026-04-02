r"""Script to train and evaluate simple baseline models."""

import itertools
import numpy as np
import wandb
import xarray as xr

from dawgz import job, schedule
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

# isort: split
from benthic.config import FOLDER_MODELS, PATH_DATASET_TRAIN_VAL
from benthic.data.const import DATASET_INPUT_VARIABLES, DATASET_OUTPUT_VARIABLES
from benthic.data.dataset import get_datasets
from benthic.data.tools import Statistics, from_zarr_to_numpy

# fmt: off
#
# Folder to save baseline outputs
FOLDER_BASELINES = FOLDER_MODELS / "baselines"

CONFIG = {
    "wandb": {
        "team":    "benthicAI",
        "project": "BenthicAI - Baselines",
        "mode":    "online",
    },
    "dataset": {
        "r_validation":    3,
        "k_validation":    10,
        "k_test":          5,
        "seed_validation": 1,
        "seed_test":       1,
    },
    "cluster": {
        "account":   "bsmfc",
        "cpus":      1,
        "ram":       "1GB",
        "partition": "batch",
        "time":      "00:05:00",
    },
}

MODELS = {
    "linear_regression": {
        "estimator_class": LinearRegression,
        "param_grid": {
            "fit_intercept": [True, False],
        },
    },
    "ridge": {
        "estimator_class": Ridge,
        "param_grid": {
            "alpha":         np.logspace(-3, 3, 5).tolist(),
            "fit_intercept": [True, False],
        },
    },
    "knn": {
        "estimator_class": KNeighborsRegressor,
        "param_grid": {
            "n_neighbors": [2, 5, 10, 15, 20, 30],
            "weights":     ["uniform", "distance"],
        },
    },
    "random_forest": {
        "estimator_class": RandomForestRegressor,
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10],
        },
    },
}

def expand_param_grid(param_grid: dict) -> list[dict]:
    r"""Return all combinations from a sklearn-style param_grid dict."""
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]


def baseline(fold_idx: int, method: str, estimator_class, params: dict, run_id: str, model_folder):
    r"""Train and evaluate a baseline model on a single fold.

    Arguments:
        fold_idx:        Index of the fold for k-fold cross-validation.
        method:          Name of the baseline method (used as W&B filter tag).
        estimator_class: Unfitted sklearn estimator class.
        params:          Hyperparameter dict passed to estimator_class(**params).
        run_id:          Unique identifier shared across all folds of this config.
        model_folder:    Path to the folder where outputs are saved.
    """
    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)

    X, y = (
        from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES),
        from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES),
    )

    X_train, y_train, X_val, y_val, _, _, _, coords_val, _ = get_datasets(X, y, **CONFIG["dataset"])

    X_train, y_train, X_val, y_val, coords_v = (
        X_train[fold_idx],
        y_train[fold_idx],
        X_val[fold_idx],
        y_val[fold_idx],
        coords_val[fold_idx],
    )

    stats_X, stats_y = Statistics(), Statistics()
    stats_X.fit(X_train)
    stats_y.fit(y_train)
    X_train_s = stats_X.standardize(X_train)
    y_train_s = stats_y.standardize(y_train)
    X_val_s   = stats_X.standardize(X_val)

    estimator = estimator_class(**params)
    estimator.fit(X_train_s, y_train_s)

    y_pred = stats_y.unstandardize(estimator.predict(X_val_s))
    y_true = y_val

    np.save(model_folder / f"{fold_idx}_coords.npy", coords_v)
    np.save(model_folder / f"{fold_idx}_y_true.npy",  y_true)
    np.save(model_folder / f"{fold_idx}_y_pred.npy",  y_pred)

    rmse_per_output = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    wandb.init(
        mode=CONFIG["wandb"]["mode"],
        project=CONFIG["wandb"]["project"],
        entity=CONFIG["wandb"]["team"],
        group=run_id,
        name=f"{run_id}_{fold_idx}",
        config={"method": method, "fold": fold_idx, **params},
    )
    wandb.log({"RMSE / Mean": rmse_per_output.mean(), "RMSE / Std": rmse_per_output.std()})
    wandb.finish()


if __name__ == "__main__":
    num_folds = CONFIG["dataset"]["r_validation"] * CONFIG["dataset"]["k_validation"]
    all_jobs  = []

    for method, model_config in MODELS.items():
        estimator_class = model_config["estimator_class"]

        for params in expand_param_grid(model_config["param_grid"]):

            run_id       = wandb.util.generate_id()
            model_folder = FOLDER_BASELINES / f"{method}_{run_id}"
            if not model_folder.exists():
                model_folder.mkdir(parents=True)

            @job(array=num_folds, **CONFIG["cluster"])
            def train(
                i:               int,
                run_id:          str  = run_id,
                method:          str  = method,
                estimator_class       = estimator_class,
                params:          dict = params,
                model_folder          = model_folder,
            ):
                baseline(
                    fold_idx        = i,
                    method          = method,
                    estimator_class = estimator_class,
                    params          = params,
                    run_id          = run_id,
                    model_folder    = model_folder,
                )

            all_jobs.append(train)

    schedule(
        *all_jobs,
        name="BENTHIC-BASELINE",
        backend="slurm",
        export="ALL",
    )
