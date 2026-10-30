r"""Script to check the dataset as the training sees it (no training).

Checks that
1. the outputs are the expected trait groups and have the expected structure,
2. the selected input features are not too correlated,
3. stations with identical inputs never end up in two different folds.
"""

import argparse
import numpy as np
import xarray as xr

# isort: split
from benthic.config import PATH_DATASET_TRAIN_VAL
from benthic.data.const import (
    DATASET_INPUT_FEATURES,
    DATASET_INPUT_VARIABLES,
    DATASET_OUTPUT_VARIABLES,
    OUTPUT_EFFECT_GROUPS,
    OUTPUT_INDEPENDENT_VARIABLES,
    OUTPUT_PROPORTION_GROUPS,
)
from benthic.data.dataset import get_datasets, group_kfold, input_groups
from benthic.data.tools import (
    check_output_groups,
    from_zarr_to_numpy,
    input_feature_names,
    select_features,
    warn_correlated_features,
)


def check_outputs(ds: xr.Dataset, y: np.ndarray):
    r"""Print how the outputs are grouped and check their structure."""
    duplicated = {v for v in DATASET_OUTPUT_VARIABLES if DATASET_OUTPUT_VARIABLES.count(v) > 1}
    if duplicated:
        raise ValueError(f"Outputs listed twice: {sorted(duplicated)}")

    print(f"{len(DATASET_OUTPUT_VARIABLES)} outputs")
    for kind, groups in (("proportion", OUTPUT_PROPORTION_GROUPS), ("effect", OUTPUT_EFFECT_GROUPS)):
        for name, modalities in groups.items():
            print(f"  {kind:10s} {name:28s} {len(modalities)} modalities")
    print(f"  independent {', '.join(OUTPUT_INDEPENDENT_VARIABLES)}")

    # Variables of the dataset used neither as input nor as output, to spot a forgotten one
    unused = sorted(set(ds.data_vars) - set(DATASET_INPUT_VARIABLES) - set(DATASET_OUTPUT_VARIABLES))
    print(f"  not used: {', '.join(unused) if unused else 'none'}")

    total = check_output_groups(y)
    print(f"  OK: group sums as expected (effect total from {total.min():.2f} to {total.max():.2f})")


def check_inputs(X: np.ndarray, names: list, threshold: float):
    r"""Print the strongest correlation between selected features and warn above threshold."""
    corr = np.corrcoef(np.unique(X, axis=0), rowvar=False)
    upper = np.abs(np.triu(corr, k=1))
    i, j = np.unravel_index(np.nanargmax(upper), upper.shape)
    print(f"\n{len(names)} input features: {', '.join(names)}")
    print(f"  strongest pair: {names[i]}/{names[j]} (r = {corr[i, j]:+.2f})")
    if not warn_correlated_features(X, names, threshold=threshold):
        print(f"  OK: no pair with |r| >= {threshold}")


def shared_rows(A: np.ndarray, B: np.ndarray) -> int:
    r"""Number of rows of A that also appear in B."""
    rows_B = {row.tobytes() for row in np.ascontiguousarray(B)}
    return sum(row.tobytes() in rows_B for row in np.ascontiguousarray(A))


def check_folds(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict):
    r"""Check that no input vector is shared between test folds, or between train and val/test."""
    n_groups = len(np.unique(groups))
    print(f"\n{len(X)} stations in {n_groups} groups of identical inputs (largest has {np.bincount(groups).max()})")

    X_train, _, X_val, _, X_test, _, _, _, _ = get_datasets(X, y, groups=groups, **cfg)
    problems = 0

    # Outer split: test folds are disjoint and cover every station
    sizes = [len(fold) for fold in X_test]
    print(f"  test folds: {sizes} stations")
    if sum(sizes) != len(X):
        print(f"  ERROR: test folds hold {sum(sizes)} stations instead of {len(X)}")
        problems += 1
    for a in range(len(X_test)):
        for b in range(a + 1, len(X_test)):
            if (n := shared_rows(X_test[a], X_test[b])):
                print(f"  ERROR: test folds {a} and {b} share {n} input vectors")
                problems += 1

    # Inner split: train and val are disjoint, and both stay outside test fold 0
    for k, (X_tr, X_va) in enumerate(zip(X_train, X_val)):
        for label, n in (
            ("train/val", shared_rows(X_va, X_tr)),
            ("train/test", shared_rows(X_test[0], X_tr)),
            ("val/test", shared_rows(X_test[0], X_va)),
        ):
            if n:
                print(f"  ERROR: inner fold {k} shares {n} input vectors between {label}")
                problems += 1
        print(f"  inner fold {k}: {len(X_tr)} train, {len(X_va)} val")

    # Without groups, the same split would leak: shows that the check can actually fail
    leaks = sum(shared_rows(X[va], X[tr]) for tr, va in group_kfold(np.arange(len(X)), cfg["k_test"], seed=0))
    print(f"  (for comparison, an ungrouped {cfg['k_test']}-fold split leaks {leaks} stations)")

    if problems:
        raise ValueError(f"{problems} problems in the folds.")
    print("  OK: identical inputs always stay in the same fold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check outputs, input correlation and fold grouping.")
    parser.add_argument("--stats", nargs="+", default=["mean", "std", "min", "max"], help="Temporal statistics.")
    parser.add_argument("--period", type=str, default="annual", help="annual, seasonal or monthly.")
    parser.add_argument("--features", nargs="+", default=DATASET_INPUT_FEATURES, help='Input features, or "all".')
    parser.add_argument("--threshold", "-t", type=float, default=0.9, help="|r| above which a pair is redundant.")
    parser.add_argument("--k_validation", type=int, default=2)
    parser.add_argument("--k_test", type=int, default=4)
    parser.add_argument("--r_validation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1, help="Seed of both the test and validation splits.")
    args = parser.parse_args()

    # Same steps as training() up to the split
    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)
    X = from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES, stats=args.stats, period=args.period)
    y = from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES)

    names = input_feature_names(ds, DATASET_INPUT_VARIABLES, stats=args.stats, period=args.period)
    if args.features != ["all"]:
        X, names = select_features(X, names, args.features), args.features

    check_outputs(ds, y)
    check_inputs(X, names, args.threshold)
    check_folds(
        X,
        y,
        input_groups(X),
        dict(
            r_validation=args.r_validation,
            k_validation=args.k_validation,
            k_test=args.k_test,
            seed_validation=args.seed,
            seed_test=args.seed,
        ),
    )
