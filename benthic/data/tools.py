r"""Tools for data manipulation."""

import numpy as np
import xarray as xr

from benthic.data.const import (
    DATASET_OUTPUT_VARIABLES,
    MONTHLY_INDICES,
    OUTPUT_EFFECT_GROUPS,
    OUTPUT_INDEPENDENT_VARIABLES,
    OUTPUT_PROPORTION_GROUPS,
    SEASONAL_INDICES,
)
from typing import List


class Statistics:
    r"""Standardization utility for numpy arrays."""

    def __init__(self):
        r"""Initialize Statistics object."""
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "Statistics":
        r"""Compute mean and standard deviation."""
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def standardize(self, X: np.ndarray) -> np.ndarray:
        r"""Standardize data using stored mean and std."""
        if self.mean is None or self.std is None:
            raise ValueError("Statistics not fitted. Call fit() first.")
        return (X - self.mean) / self.std

    def unstandardize(self, X: np.ndarray) -> np.ndarray:
        r"""Unstandardize data using stored mean and std."""
        if self.mean is None or self.std is None:
            raise ValueError("Statistics not fitted. Call fit() first.")
        return X * self.std + self.mean


def extract_temporal_statistics(
    ds: xr.DataArray,
    stats: List[str] | None = None,
    period: str = "annual",
) -> np.ndarray:
    r"""Extract temporal statistics from a time-dependent DataArray.

    Arguments:
        ds:     DataArray with shape (station, time) containing temporal data.
        stats:  List of statistics to compute. Supported: ["mean", "std", "min", "max"].
        period: Temporal granularity for statistics computation. One of "annual", "monthly", or "seasonal".
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]

    values = ds.values

    if period == "annual":
        periods_dict = {"annual": list(range(values.shape[1]))}
    elif period == "monthly":
        periods_dict = MONTHLY_INDICES
    elif period == "seasonal":
        periods_dict = SEASONAL_INDICES
    else:
        raise ValueError(
            f"Unsupported period: {period!r}. Choose from 'annual', 'monthly', 'seasonal'."
        )

    period_blocks = []
    for t_indices in periods_dict.values():
        subset = values[:, t_indices]
        stat_arrays = []
        for stat in stats:
            if stat == "mean":
                stat_arrays.append(np.nanmean(subset, axis=1, keepdims=True))
            elif stat == "std":
                stat_arrays.append(np.nanstd(subset, axis=1, keepdims=True))
            elif stat == "min":
                stat_arrays.append(np.nanmin(subset, axis=1, keepdims=True))
            elif stat == "max":
                stat_arrays.append(np.nanmax(subset, axis=1, keepdims=True))
            else:
                raise ValueError(
                    f"Unsupported statistic: {stat!r}. Choose from 'mean', 'std', 'min', 'max'."
                )
        period_blocks.append(np.concatenate(stat_arrays, axis=1))

    # (station, n_periods * n_stats)
    return np.concatenate(period_blocks, axis=1)


def from_zarr_to_numpy(
    ds: xr.Dataset,
    variables: List[str],
    stats: List[str] | None = None,
    period: str = "annual",
) -> np.ndarray:
    r"""Transform an xarray Dataset into a numpy feature matrix.

    Arguments:
        ds:        xarray Dataset containing the variables to transform.
        variables: List of variable names to extract from the dataset.
        stats:     List of statistics to compute for time-dependent variables.
        period:    Temporal granularity for statistics computation. One of "annual",
                   "monthly" (12 blocks), or "seasonal" (4 blocks).
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]

    features_list = []
    for var_name in variables:
        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found in dataset")

        da = ds[var_name]

        # Check if the variable has a temporal dimension
        if "time" in da.dims:
            stats_array = extract_temporal_statistics(da, stats, period)
            features_list.append(stats_array)
        else:
            static_array = da.values.reshape(-1, 1)
            features_list.append(static_array)

    # Concatenate all features horizontally
    return np.concatenate(features_list, axis=1)


def filter_outliers(
    X: np.ndarray,
    q_min: float = 0.0,
    q_max: float = 100.0,
    zero_safe: bool = True,
) -> np.ndarray:
    r"""Clip per-column outliers at given percentile bounds.

    Arguments:
        X:         Feature matrix of shape (n_samples, n_features).
        q_min:     Lower percentile for clipping (0 = no lower clipping).
        q_max:     Upper percentile for clipping (100 = no upper clipping).
        zero_safe: If True, the lower bound is floored at 0 so that physically meaningful zeros on R+ variables are never removed.
    """
    if q_min == 0.0 and q_max == 100.0:
        return X

    X_out = X.copy()
    for j in range(X.shape[1]):
        col = X_out[:, j]
        lower = float(np.nanpercentile(col, q_min)) if q_min > 0 else -np.inf
        upper = float(np.nanpercentile(col, q_max)) if q_max < 100 else np.inf
        if zero_safe and lower > 0:
            lower = 0.0
        X_out[:, j] = np.clip(col, lower, upper)

    return X_out

def check_output_groups(y: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    r"""Check that the output trait groups have the expected structure.
 
    Proportion groups must sum to 1, every effect group of a station must sum to the
    same total, and independent outputs must lie in [0, 1].
 
    Arguments:
        y:   Output matrix of shape (n_samples, n_outputs), ordered as DATASET_OUTPUT_VARIABLES.
        tol: Tolerance on the sums (absolute for proportions, relative for effect totals).
 
    Returns:
        total: Station total log-abundance of the effect groups, shape (n_samples,).
    """
    index = {name: i for i, name in enumerate(DATASET_OUTPUT_VARIABLES)}
 
    if np.isnan(y).any():
        raise ValueError(f"Outputs contain NaN values in {np.isnan(y).any(axis=0).sum()} columns.")
 
    # Proportion groups (Resp_mod, POSE_grp) sum to 1
    for name, modalities in OUTPUT_PROPORTION_GROUPS.items():
        sums = y[:, [index[m] for m in modalities]].sum(axis=1)
        bad = np.abs(sums - 1) > tol
        if bad.any():
            raise ValueError(
                f"Group '{name}' does not sum to 1 at {bad.sum()} stations "
                f"(range {sums.min():.3f} to {sums.max():.3f})."
            )
 
    # Effect groups (Effect_mod) all sum to the same station total
    effect_sums = np.stack(
        [y[:, [index[m] for m in modalities]].sum(axis=1) for modalities in OUTPUT_EFFECT_GROUPS.values()],
        axis=1,
    )
    total = effect_sums.mean(axis=1)
    if (total <= 0).any():
        raise ValueError(f"{(total <= 0).sum()} stations have a null total abundance.")
 
    gap = np.abs(effect_sums - total[:, None]) / total[:, None]
    for j, name in enumerate(OUTPUT_EFFECT_GROUPS):
        bad = gap[:, j] > tol
        if bad.any():
            raise ValueError(
                f"Group '{name}' does not sum to the station total at {bad.sum()} stations "
                f"(max relative gap {gap[:, j].max():.3f})."
            )
 
    # Independent outputs (ind_ecol) lie in [0, 1]
    for name in OUTPUT_INDEPENDENT_VARIABLES:
        values = y[:, index[name]]
        if values.min() < -tol or values.max() > 1 + tol:
            raise ValueError(f"'{name}' is outside [0, 1] (range {values.min():.3f} to {values.max():.3f}).")
 
    return total
