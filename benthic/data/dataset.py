r"""Datasets."""

import numpy as np
import torch

from benthic.config import PATH_DATASET_COORDINATES
from sklearn.model_selection import KFold
from typing import List, Optional, Tuple


def get_datasets(
    X: np.ndarray,
    y: np.ndarray,
    r_validation: int,
    k_validation: int,
    k_test: int,
    seed_validation: Optional[int] = None,
    seed_test: Optional[int] = None,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    r"""Create nested k-fold splits for double cross-validation.

    Arguments:
        X: Input features of shape (n_samples, n_features).
        y: Target values of shape (n_samples, n_targets).
        r_validation: Number of repetitions for validation folds.
        k_validation: Number of folds for validation splits.
        k_test: Number of folds for test splits.
        seed_validation: Random seed for validation splits. If None, splits are random.
        seed_test: Random seed for test splits. If None, splits are random.

    Returns:
        X_train:      List of (r_validation * k_validation) training input folds
        y_train:      List of (r_validation * k_validation) training target folds
        X_val:        List of (r_validation * k_validation) validation input folds
        y_val:        List of (r_validation * k_validation) validation target folds
        X_test:       List of k_test test input folds
        y_test:       List of k_test test target folds
        coords_train: List of (r_validation * k_validation) training coordinate folds, each (n, 3)
        coords_val:   List of (r_validation * k_validation) validation coordinate folds, each (n, 3)
        coords_test:  List of k_test test coordinate folds, each (n, 3)
    """
    # Loading coordinates of stations (STATIONS, 3) with lat, lon, depth
    coordinates = torch.load(PATH_DATASET_COORDINATES, weights_only=True).numpy()

    # Create k_test folds on all data
    kfold_test = KFold(n_splits=k_test, shuffle=True, random_state=seed_test)

    X_test, y_test, coords_test, test_indices_list = [], [], [], []

    for _, test_idx in kfold_test.split(X):
        X_test.append(X[test_idx])
        y_test.append(y[test_idx])
        coords_test.append(coordinates[test_idx])
        test_indices_list.append(test_idx)

    # Use remaining data (all folds except the first) for train/val
    remaining_indices = np.concatenate(test_indices_list[1:])
    X_remaining = X[remaining_indices]
    y_remaining = y[remaining_indices]

    # Create r_validation repetitions of k_validation folds
    X_train, X_val, y_train, y_val, coords_train, coords_val = [], [], [], [], [], []

    for r in range(r_validation):
        # Use different seed for each repetition if seed_validation is provided
        current_seed = seed_validation + r if seed_validation is not None else None

        # Create k_validation folds on the remaining data
        kfold_val = KFold(n_splits=k_validation, shuffle=True, random_state=current_seed)

        for train_idx, val_idx in kfold_val.split(X_remaining):
            X_train.append(X_remaining[train_idx])
            X_val.append(X_remaining[val_idx])
            y_train.append(y_remaining[train_idx])
            y_val.append(y_remaining[val_idx])
            coords_train.append(coordinates[remaining_indices[train_idx]])
            coords_val.append(coordinates[remaining_indices[val_idx]])

    return X_train, y_train, X_val, y_val, X_test, y_test, coords_train, coords_val, coords_test
