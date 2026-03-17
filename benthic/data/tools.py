r"""Tools for data manipulation."""

import numpy as np
import xarray as xr

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


def extract_temporal_statistics(ds: xr.DataArray, stats: List[str] | None = None) -> np.ndarray:
    r"""Extract temporal statistics from a time-dependent DataArray.

    Arguments:
        ds: DataArray with shape (station, time) containing temporal data.
        stats: List of statistics to compute (supported values ["mean", "std", "min", "max"]).
    """
    if stats is None:
        stats = ["mean", "std", "min", "max"]

    # Extract numpy values (station, time)
    values = ds.values

    # Compute each statistic along the temporal axis (axis=1)
    stat_arrays = []
    for stat in stats:
        if stat == "mean":
            stat_arrays.append(np.nanmean(values, axis=1, keepdims=True))
        elif stat == "std":
            stat_arrays.append(np.nanstd(values, axis=1, keepdims=True))
        elif stat == "min":
            stat_arrays.append(np.nanmin(values, axis=1, keepdims=True))
        elif stat == "max":
            stat_arrays.append(np.nanmax(values, axis=1, keepdims=True))
        else:
            raise ValueError(f"Unsupported statistic: {stat}")

    # Concatenate all statistics horizontally: (station, n_stats)
    return np.concatenate(stat_arrays, axis=1)


def from_zarr_to_numpy(
    ds: xr.Dataset, variables: List[str], stats: List[str] | None = None
) -> np.ndarray:
    r"""Transform an xarray Dataset into a numpy feature matrix.

    Arguments:
        ds: xarray Dataset containing the variables to transform.
        variables: List of variable names to extract from the dataset.
        stats: List of statistics to compute for time-dependent variables.
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
            stats_array = extract_temporal_statistics(da, stats)
            features_list.append(stats_array)
        else:
            static_array = da.values.reshape(-1, 1)  # (n_stations, 1)
            features_list.append(static_array)

    # Concatenate all features horizontally
    return np.concatenate(features_list, axis=1)
