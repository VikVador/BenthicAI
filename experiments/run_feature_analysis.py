r"""Script to measure how redundant the input features are (no training)."""

import argparse
import numpy as np
import pandas as pd
import xarray as xr

# isort: split
from benthic.config import PATH_DATASET_TRAIN_VAL
from benthic.data.const import DATASET_INPUT_VARIABLES
from benthic.data.dataset import input_groups
from benthic.data.tools import from_zarr_to_numpy

STATS = ["mean", "std", "min", "max"]


def feature_names(ds: xr.Dataset, variables: list, stats: list) -> list:
    r"""Names of the columns built by from_zarr_to_numpy with period="annual"."""
    names = []
    for var in variables:
        if "time" in ds[var].dims:
            names += [f"{var}_{s}" for s in stats]
        else:
            names.append(var)
    return names


def one_row_per_group(X: np.ndarray) -> np.ndarray:
    r"""Keep one station per group of identical inputs, so duplicates don't weigh more."""
    _, first = np.unique(input_groups(X), return_index=True)
    return X[np.sort(first)]


def prune_correlated(corr: pd.DataFrame, threshold: float) -> tuple[list, list]:
    r"""Greedily drop features until no pair has |r| >= threshold.

    At each step, among the features involved in a too-strong pair, the one with the
    highest mean |r| to all other features is dropped (it is the most redundant).

    Returns:
        kept:    Features that remain.
        dropped: (feature, partner with highest |r|, r) in the order they were dropped.
    """
    kept, dropped = list(corr.columns), []
    while True:
        sub = corr.loc[kept, kept].abs().where(~np.eye(len(kept), dtype=bool))
        if sub.max().max() < threshold:
            return kept, dropped
        candidates = sub.columns[(sub >= threshold).any()]
        worst = sub[candidates].mean().idxmax()
        partner = sub[worst].idxmax()
        dropped.append((worst, partner, round(float(corr.loc[worst, partner]), 2)))
        kept.remove(worst)


def pca_variance(Z: np.ndarray) -> np.ndarray:
    r"""Cumulative fraction of variance explained by the principal components of Z."""
    s = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    return np.cumsum(s**2) / np.sum(s**2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure redundancy among input features.")
    parser.add_argument("--threshold", "-t", type=float, default=0.9, help="|r| above which a pair is redundant.")
    parser.add_argument("--output", "-o", type=str, default="features", help="Prefix of the CSV outputs.")
    args = parser.parse_args()

    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)
    X = from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES, stats=STATS, period="annual")
    names = feature_names(ds, DATASET_INPUT_VARIABLES, STATS)
    if len(names) != X.shape[1]:
        raise ValueError(f"{len(names)} names for {X.shape[1]} columns.")

    X = one_row_per_group(X)
    Z = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    if np.isnan(Z).any():
        print(f"Warning: {np.isnan(Z).any(axis=0).sum()} features contain NaN, set to 0 after standardizing.")
        Z = np.nan_to_num(Z)

    corr = pd.DataFrame(np.corrcoef(Z, rowvar=False), index=names, columns=names)

    # 1. Strongest pairs
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
    print(f"{X.shape[1]} features on {X.shape[0]} distinct input vectors\n")
    print(f"Pairs with |r| >= {args.threshold}: {(pairs.abs() >= args.threshold).sum()}")
    print(pairs[pairs.abs() >= args.threshold].round(2).to_string())

    # 2. How similar the 4 statistics of each variable are
    print("\nCorrelation between the statistics of each variable (mean vs std / min / max)")
    for var in DATASET_INPUT_VARIABLES:
        if f"{var}_mean" in names:
            row = {s: corr.loc[f"{var}_mean", f"{var}_{s}"] for s in STATS[1:]}
            print(f"  {var:16s}" + "".join(f"  {s} {r:+.2f}" for s, r in row.items()))

    # 3. Effective dimension
    cumulative = pca_variance(Z)
    print("\nPrincipal components needed to explain the variance of the inputs")
    for level in (0.90, 0.95, 0.99):
        print(f"  {level:.0%}: {int(np.searchsorted(cumulative, level) + 1)} components")

    # 4. A suggested reduced set
    kept, dropped = prune_correlated(corr, args.threshold)
    print(f"\nGreedy pruning at |r| >= {args.threshold}: keeps {len(kept)} of {len(names)}")
    for feature, partner, r in dropped:
        print(f"  drop {feature:22s} (r = {r:+.2f} with {partner})")
    print("  keep " + ", ".join(kept))

    corr.to_csv(f"{args.output}_correlation.csv")
    pd.Series(kept, name="feature").to_csv(f"{args.output}_kept.csv", index=False)
