r"""Datasets."""

import numpy as np
import torch

from benthic.config import PATH_DATASET_COORDINATES
from typing import List, Optional, Tuple


def input_groups(X: np.ndarray) -> np.ndarray:
    r"""Label stations that share exactly the same input vector.

    Stations in the same model grid cell (with the same sediment values) have identical
    inputs. They must stay together when splitting, otherwise validation sees inputs
    already seen in training.

    Arguments:
        X: Input features of shape (n_samples, n_features).

    Returns:
        groups: Integer group label per station, shape (n_samples,).
    """
    # Comparing raw bytes so that rows with NaN at the same place still match
    labels = {}
    return np.array([labels.setdefault(row.tobytes(), len(labels)) for row in np.ascontiguousarray(X)])


def group_kfold(
    groups: np.ndarray,
    n_splits: int,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    r"""Shuffled k-fold split that keeps every group in a single fold.

    Groups are shuffled, then each group goes to the fold with the fewest stations so far,
    so folds stay balanced even with groups of different sizes. Unlike sklearn's GroupKFold
    before 1.6, a different seed gives a different split.

    Arguments:
        groups:   Group label per station, shape (n_samples,).
        n_splits: Number of folds.
        seed:     Random seed for shuffling the groups. If None, splits are random.

    Returns:
        List of n_splits (train_idx, val_idx) pairs of station indices.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Cannot make {n_splits} folds from {len(unique_groups)} groups.")

    rng = np.random.default_rng(seed)
    fold_of_group, fold_sizes = {}, np.zeros(n_splits, dtype=int)

    for g in rng.permutation(unique_groups):
        fold = int(np.argmin(fold_sizes))
        fold_of_group[g] = fold
        fold_sizes[fold] += int((groups == g).sum())

    fold_of_station = np.array([fold_of_group[g] for g in groups])

    return [
        (np.where(fold_of_station != k)[0], np.where(fold_of_station == k)[0])
        for k in range(n_splits)
    ]


def get_datasets(
    X: np.ndarray,
    y: np.ndarray,
    r_validation: int,
    k_validation: int,
    k_test: int,
    seed_validation: Optional[int] = None,
    seed_test: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
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
    r"""Create nested grouped k-fold splits for double cross-validation.

    Stations with identical inputs (same group) always end up in the same fold, both in the
    outer test split and in the inner validation splits.

    Arguments:
        X: Input features of shape (n_samples, n_features).
        y: Target values of shape (n_samples, n_targets).
        r_validation: Number of repetitions for validation folds.
        k_validation: Number of folds for validation splits.
        k_test: Number of folds for test splits.
        seed_validation: Random seed for validation splits. If None, splits are random.
        seed_test: Random seed for test splits. If None, splits are random.
        groups: Group label per station. If None, computed from identical rows of X.

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

    # Stations with identical inputs must never be split across folds
    if groups is None:
        groups = input_groups(X)

    # Create k_test grouped folds on all data
    X_test, y_test, coords_test, test_indices_list = [], [], [], []

    for _, test_idx in group_kfold(groups, k_test, seed=seed_test):
        X_test.append(X[test_idx])
        y_test.append(y[test_idx])
        coords_test.append(coordinates[test_idx])
        test_indices_list.append(test_idx)

    # Use remaining data (all folds except the first) for train/val
    remaining_indices = np.concatenate(test_indices_list[1:])
    X_remaining = X[remaining_indices]
    y_remaining = y[remaining_indices]
    groups_remaining = groups[remaining_indices]

    # Create r_validation repetitions of k_validation grouped folds
    X_train, X_val, y_train, y_val, coords_train, coords_val = [], [], [], [], [], []

    for r in range(r_validation):
        # Use different seed for each repetition if seed_validation is provided
        current_seed = seed_validation + r if seed_validation is not None else None

        for train_idx, val_idx in group_kfold(groups_remaining, k_validation, seed=current_seed):
            X_train.append(X_remaining[train_idx])
            X_val.append(X_remaining[val_idx])
            y_train.append(y_remaining[train_idx])
            y_val.append(y_remaining[val_idx])
            coords_train.append(coordinates[remaining_indices[train_idx]])
            coords_val.append(coordinates[remaining_indices[val_idx]])

    return X_train, y_train, X_val, y_val, X_test, y_test, coords_train, coords_val, coords_test
