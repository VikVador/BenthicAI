r"""Script to compute the R^2 ceiling of every output and every trait (no training)."""

import argparse
import numpy as np
import pandas as pd
import xarray as xr

# isort: split
from benthic.config import PATH_DATASET_TRAIN_VAL
from benthic.data.const import (
    DATASET_INPUT_VARIABLES,
    DATASET_OUTPUT_VARIABLES,
    OUTPUT_EFFECT_GROUPS,
    OUTPUT_INDEPENDENT_VARIABLES,
    OUTPUT_PROPORTION_GROUPS,
)
from benthic.data.dataset import input_groups
from benthic.data.tools import check_output_groups, from_zarr_to_numpy
from benthic.diagnostics.metrics import noise_ceiling

COLUMNS = ("sd_ratio", "r2_max", "r2_max_low", "r2_max_high")


def trait_blocks(y: np.ndarray) -> tuple[np.ndarray, dict]:
    r"""Rearrange outputs so that every trait is a block of proportions.

    Effect_mod traits are divided by the station total, which becomes its own block, so
    every trait block sums to 1 and the abundance is evaluated separately.
    """
    index = {name: i for i, name in enumerate(DATASET_OUTPUT_VARIABLES)}
    total = check_output_groups(y)

    columns, blocks, start = [], {}, 0

    def add(name: str, values: np.ndarray):
        nonlocal start
        blocks[name] = list(range(start, start + values.shape[1]))
        columns.append(values)
        start += values.shape[1]

    for groups, scale in ((OUTPUT_PROPORTION_GROUPS, 1.0), (OUTPUT_EFFECT_GROUPS, total[:, None])):
        for name, modalities in groups.items():
            add(name, y[:, [index[m] for m in modalities]] / scale)

    add("total_abundance", total[:, None])

    for name in OUTPUT_INDEPENDENT_VARIABLES:
        add(name, y[:, [index[name]]])

    return np.concatenate(columns, axis=1), blocks


def as_table(result: dict, names: list, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {k: result[k] for k in COLUMNS}, index=pd.Index(names, name=label)
    ).sort_values("r2_max")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the R^2 ceiling of every output and trait.")
    parser.add_argument("--output", "-o", type=str, default="ceiling", help="Prefix of the CSV tables.")
    parser.add_argument("--bootstrap", "-b", type=int, default=1000, help="Number of bootstrap resamples.")
    args = parser.parse_args()

    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)
    X, y = (
        from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES),
        from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES),
    )
    groups = input_groups(X)

    # One ceiling per trait (modalities as a vector, Effect_mod as proportions + total)
    z, blocks = trait_blocks(y)
    per_trait = noise_ceiling(z, groups, blocks=blocks, num_bootstrap=args.bootstrap)
    table_trait = as_table(per_trait, list(blocks), "trait")

    # One ceiling per output column, as a diagnostic of which modality is noisy
    per_output = noise_ceiling(y, groups, num_bootstrap=args.bootstrap)
    table_output = as_table(per_output, DATASET_OUTPUT_VARIABLES, "output")

    print(f"Within-group noise from {per_trait['num_groups']} groups ({per_trait['num_stations']} stations)\n")
    print("Per trait\n" + table_trait.round(2).to_string())
    print(f"\nMedian R^2 ceiling per trait: {table_trait['r2_max'].median():.2f}")
    print("\nPer output\n" + table_output.round(2).to_string())

    table_trait.to_csv(f"{args.output}_traits.csv")
    table_output.to_csv(f"{args.output}_outputs.csv")
