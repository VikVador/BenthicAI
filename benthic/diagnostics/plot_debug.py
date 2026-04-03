import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from pathlib import Path
from scipy.stats import spearmanr

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

from benthic.config import PATH_MODEL_BATHYMETRY
from benthic.data.const import DATASET_OUTPUT_VARIABLES


class WandbPlotLogger:
    def __init__(self, run, namespace="plots", use_tables=True):
        self.run = run
        self.namespace = namespace
        self.use_tables = use_tables
        self.tables = {}
        self.logged_once = set()

    def _get_table(self, name):
        if name not in self.tables:
            self.tables[name] = wandb.Table(columns=["variable", "image"])
        return self.tables[name]

    def log(self, plot_type, variable, fig, metric=None):
        """
        plot_type: e.g. 'distribution', 'correlation', 'map', 'error'
        variable: output variable name
        metric: optional (e.g. 'bias', 'rmse')
        """
        if metric:
            key = f"{self.namespace}/{plot_type}/{metric}/{variable}"
            table_name = f"{self.namespace}/{plot_type}/{metric}"
        else:
            key = f"{self.namespace}/{plot_type}/{variable}"
            table_name = f"{self.namespace}/{plot_type}"

        if self.use_tables:
            table = self._get_table(table_name)
            table.add_data(variable, wandb.Image(fig))
        else:
            self.run.log({key: wandb.Image(fig)})

        plt.close(fig)

    def flush(self):
        if self.use_tables:
            payload = {name: table for name, table in self.tables.items()}
            if payload:
                self.run.log(payload)
                self.tables = {}

    def log_once(self, key, value):
        if key in self.logged_once:
            return
        self.run.log({key: value})
        self.logged_once.add(key)


def plot_error_per_station(values, title=None, ylabel="Error", figsize=(12, 4), show=False):
    """
    Plot a barplot of error values per station.

    Parameters
    ----------
    values : array-like, shape (n_stations,)
        Array of values (e.g., bias or RMSE).
    title : str, optional
        Plot title.
    ylabel : str, optional
        Label for y-axis.
    figsize : tuple, optional
        Figure size.
    """
    values = np.asarray(values)
    n = len(values)

    x = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)

    # Color bars differently for positive/negative (useful for bias)
    colors = np.where(values >= 0, "tab:blue", "tab:red")

    ax.bar(x, values, color=colors)

    # Reference zero line (important for bias)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_xlabel("Stations")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    # Optional: avoid overcrowded x ticks for many stations
    if n <= 50:
        ax.set_xticks(x)
    else:
        ax.set_xticks([])

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_data_distribution(
    y_true,
    y_pred,
    bins=30,
    title=None,
    xlabel="Value",
    ylabel="Count",
    figsize=(10, 6),
    show=False,
):
    """
    Plot overlapping histograms of y_true and y_pred using identical bins.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    bins : int or array-like
        Number of bins or explicit bin edges.
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional
    figsize : tuple, optional
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Compute common bin edges
    data_min = min(y_true.min(), y_pred.min())
    data_max = max(y_true.max(), y_pred.max())
    bin_edges = np.linspace(data_min, data_max, bins + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(y_true, bins=bin_edges, density=True, alpha=0.5, label="y_true", color="tab:green")
    ax.hist(y_pred, bins=bin_edges, density=True, alpha=0.5, label="y_pred", color="tab:blue")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_data_correlation(
    y_true,
    y_pred,
    title=None,
    xlabel="y_true",
    ylabel="y_pred",
    figsize=(6, 6),
    show=False,
):
    """
    Plot a scatter plot of y_true vs y_pred to visualize correlation.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional
    figsize : tuple, optional
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, s=8, color="tab:purple", edgecolors="none")

    r = np.corrcoef(y_true, y_pred)[0, 1]
    spearman = spearmanr(y_true, y_pred).correlation

    vmin = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    vmax = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="black", lw=1)

    ax.set_title(f"{title}\nr={r:.2f}\nspearman={spearman:.2f}", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(["y_true", "y_pred"], loc="upper left", frameon=True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def produce_metrics_table(df, metrics, extra_df=None):
    """
    Produce a summary table of metrics for each ecological trait.

    Rows are metrics.
    Columns are DATASET_OUTPUT_VARIABLES plus a final 'tot' column,
    where 'tot' is the average across variables for that metric.

    If extra_df is provided, it is merged on coords (lat, lon, depth)
    before extracting metric columns. This allows combining metrics
    coming from different averaged dataframes (e.g., r2).
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    merged_df = df.copy()
    if extra_df is not None:
        merged_df = merged_df.merge(
            extra_df,
            on=["lat", "lon", "depth"],
            how="left",
            suffixes=("", "_extra"),
        )

    table = {}

    for metric in metrics:
        metric_values = {}

        for variable in DATASET_OUTPUT_VARIABLES:
            column_name = f"{metric}_{variable}"
            if column_name in merged_df.columns:
                metric_values[variable] = float(np.nanmean(merged_df[column_name].to_numpy()))
            else:
                metric_values[variable] = np.nan

        metric_values["tot"] = float(np.nanmean(list(metric_values.values())))
        table[str(metric)] = metric_values

    return pd.DataFrame.from_dict(table, orient="index")


def plot_maps(
    lat,
    lon,
    values,
    output_name,
    bathy_nc_path=None,
    bathy_var_name="bathymetry",
    zoom_in=(26.5, 32.0, 41.95, 47.0),
    bathy_contours=(25, 50, 100, 150),
    cmap="viridis",
    vmin=None,
    vmax=None,
    show_coastline=True,
    coastline_level=0.0,
    coastline_color="black",
    coastline_linewidth=1.0,
    show=False,
):
    """Plot full and zoomed maps for a single variable using lat/lon and values vectors.

    If bathymetry contains NaNs, coastline is drawn from the NaN boundary (sea/land mask).
    Otherwise, coastline is drawn from the bathymetry contour at coastline_level.
    """
    lat = np.asarray(lat).ravel()
    lon = np.asarray(lon).ravel()
    values = np.asarray(values).ravel()

    if not (len(lat) == len(lon) == len(values)):
        raise ValueError(
            f"lat, lon, and values must have equal length. "
            f"Got {len(lat)}, {len(lon)}, {len(values)}"
        )

    #output_dir = Path(output_dir)
    #output_dir.mkdir(parents=True, exist_ok=True)

    bathy_ds = None
    bathy_lon = None
    bathy_lat = None
    bathy_values = None
    bathy_has_nan = False

    if bathy_nc_path is not None:
        bathy_ds = xr.open_dataset(bathy_nc_path)
        if bathy_var_name not in bathy_ds:
            raise ValueError(f"Variable '{bathy_var_name}' not found in {bathy_nc_path}")
        if "lat" not in bathy_ds or "lon" not in bathy_ds:
            raise ValueError("Bathymetry dataset must contain 'lat' and 'lon' variables.")

        bathy_lon = bathy_ds["lon"].values
        bathy_lat = bathy_ds["lat"].values
        bathy_values = np.asarray(bathy_ds[bathy_var_name].values)

        finite_count = int(np.isfinite(bathy_values).sum())
        if finite_count == 0:
            raise ValueError(
                f"Bathymetry '{bathy_var_name}' contains only NaNs in {bathy_nc_path}."
            )

        bathy_has_nan = bool(np.isnan(bathy_values).any())

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)

    # Zoomed map (same bounds as make_map scripts)
    zoom_mask = (
        valid
        & (lon >= zoom_in[0])
        & (lon <= zoom_in[1])
        & (lat >= zoom_in[2])
        & (lat <= zoom_in[3])
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        lon[zoom_mask],
        lat[zoom_mask],
        c=values[zoom_mask],
        cmap=cmap,
        s=40,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
    )

    if bathy_ds is not None:
        ax.contour(
            bathy_lon,
            bathy_lat,
            bathy_values,
            levels=bathy_contours,
            colors="#FFFFFF",
            linewidths=0.8,
        )

        if show_coastline:
            if bathy_has_nan:
                sea_mask = np.isfinite(bathy_values).astype(float)
                ax.contour(
                    bathy_lon,
                    bathy_lat,
                    sea_mask,
                    levels=[0.5],
                    colors=coastline_color,
                    linewidths=coastline_linewidth,
                )
            else:
                ax.contour(
                    bathy_lon,
                    bathy_lat,
                    bathy_values,
                    levels=[coastline_level],
                    colors=coastline_color,
                    linewidths=coastline_linewidth,
                )

    ax.set_xlim(zoom_in[0], zoom_in[1])
    ax.set_ylim(zoom_in[2], zoom_in[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"zoom_{output_name}")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(output_name)
    ax.grid(False)

    #fig.savefig(output_dir / f"zoom_{output_name}.png", dpi=200, bbox_inches="tight")
    #plt.close(fig)
    if show:
        plt.show()
    if bathy_ds is not None:
        bathy_ds.close()
    return fig


def dataset_folder_name(path_dataset):
    parts = [part for part in Path(path_dataset).parts if part not in ("/", "")]
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


def log_figure(run, key, fig):
    run.log({key: wandb.Image(fig)})
    plt.close(fig)


# In[ ]:


def load_files(path_to_results, metric, standardized=False):
    if not path_to_results.exists():
        raise FileNotFoundError(f"Results folder does not exist: {path_to_results}")

    target_name = f"standardized_{metric}" if standardized else metric
    arrays = []

    # Use precomputed metric files: <fold>_<metric>.npy
    for file in sorted(path_to_results.glob("*.npy")):
        if file.stem.endswith(target_name):
            array = np.load(file)
            row_count = array.shape[0] if array.ndim > 0 else 1
            print(f"Loaded {file.name}: rows={row_count}, shape={array.shape}")
            arrays.append(array)

    if not arrays:
        Warning(f"No files found for metric '{target_name}' in {path_to_results}")
        return None

    return np.concatenate(arrays, axis=0)


def create_stacked_dataframe(coords, y_true, metrics):
    """Build a row-stacked DataFrame with explicit columns for each array dimension."""
    if coords is None:
        raise ValueError("'coords' is required to build the DataFrame.")
    print(y_true.shape)
    print(coords.shape)
    for metric in metrics:
        if metric is not None:
            print(f"{metric} shape: {np.asarray(metrics[metric]).shape}")
    coords = np.asarray(coords)
    y_true = np.asarray(y_true)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N, 3), got {coords.shape}")

    if y_true.ndim != 2:
        raise ValueError(f"Expected y_true to be 2D, got shape {y_true.shape}")

    n_outputs = len(DATASET_OUTPUT_VARIABLES)
    if y_true.shape[1] != n_outputs:
        raise ValueError(
            f"Expected y_true to have {n_outputs} columns, got {y_true.shape[1]}"
        )

    # Determine a common row count across per-station arrays to avoid DataFrame length mismatch.
    row_candidates = [coords.shape[0], y_true.shape[0]]
    for metric_name, metric_array in metrics.items():
        if metric_array is None:
            continue
        arr = np.asarray(metric_array)
        if arr.ndim == 2:
            row_candidates.append(arr.shape[0])
        elif arr.ndim == 1 and arr.shape[0] != n_outputs:
            # 1D arrays can be either per-station (rows) or per-variable (n_outputs).
            row_candidates.append(arr.shape[0])

    n_rows = min(row_candidates)
    if len(set(row_candidates)) > 1:
        print(
            f"[create_stacked_dataframe] Row mismatch detected {row_candidates}; "
            f"aligning all per-station arrays to n_rows={n_rows}."
        )

    coords = coords[:n_rows, :]
    y_true = y_true[:n_rows, :]

    data = {
        "lat": coords[:, 0],
        "lon": coords[:, 1],
        "depth": coords[:, 2],
    }

    for j, output_name in enumerate(DATASET_OUTPUT_VARIABLES):
        data[str(output_name)] = y_true[:, j]

    # metrics must be a dict: {"metric_name": ndarray, ...}
    for metric_name, metric_array in metrics.items():
        if metric_array is None:
            continue

        arr = np.asarray(metric_array)

        if arr.ndim == 1:
            # Case 1: one value per output variable (e.g. r2)
            if arr.shape[0] == n_outputs:
                for j, output_name in enumerate(DATASET_OUTPUT_VARIABLES):
                    data[f"{str(metric_name)}_{str(output_name)}"] = np.full(n_rows, arr[j])
            # Case 2: one value per station
            else:
                data[str(metric_name)] = arr[:n_rows]

        elif arr.ndim == 2:
            if arr.shape[1] != n_outputs:
                raise ValueError(
                    f"Expected {metric_name} to have {n_outputs} columns, got {arr.shape[1]}"
                )
            arr = arr[:n_rows, :]
            for j, output_name in enumerate(DATASET_OUTPUT_VARIABLES):
                data[f"{str(metric_name)}_{str(output_name)}"] = arr[:, j]

        else:
            raise ValueError(
                f"Expected {metric_name} to be 1D or 2D, got shape {arr.shape}"
            )

    return pd.DataFrame(data)


def average_stacked_dataframe_by_coords(stacked_df):
    """Average all numeric columns for rows that share the same coordinates."""
    coord_cols = ["lat", "lon", "depth"]
    return stacked_df.groupby(coord_cols, as_index=False).mean(numeric_only=True)


def create_averaged_r2_dataframe(coords, r2):
    """
    Create a DataFrame analogous to averaged_df for r2 metrics.

    Output has columns: lat, lon, depth, r2_<variable> for each output variable,
    then grouped/averaged by identical coordinates.
    """
    if coords is None or r2 is None:
        raise ValueError("'coords' and 'r2' are required.")

    coords = np.asarray(coords)
    r2 = np.asarray(r2)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N, 3), got {coords.shape}")

    n_outputs = len(DATASET_OUTPUT_VARIABLES)

    if r2.ndim == 1:
        if r2.size == n_outputs:
            r2_vector = r2
        elif r2.size % n_outputs == 0:
            r2_vector = r2.reshape(-1, n_outputs).mean(axis=0)
        else:
            raise ValueError(
                f"Unexpected r2 size {r2.size}; cannot reshape into (?, {n_outputs})."
            )
    elif r2.ndim == 2:
        if r2.shape[1] != n_outputs:
            raise ValueError(
                f"Expected r2 to have {n_outputs} columns, got {r2.shape[1]}"
            )
        r2_vector = r2.mean(axis=0)
    else:
        raise ValueError(f"Expected r2 to be 1D or 2D, got shape {r2.shape}")

    n_rows = coords.shape[0]
    data = {
        "lat": coords[:, 0],
        "lon": coords[:, 1],
        "depth": coords[:, 2],
    }

    for j, output_name in enumerate(DATASET_OUTPUT_VARIABLES):
        data[f"r2_{output_name}"] = np.full(n_rows, r2_vector[j])

    r2_df = pd.DataFrame(data)
    return r2_df.groupby(["lat", "lon", "depth"], as_index=False).mean(numeric_only=True)


# In[54]:


def select_column(averaged_df, metric, index):
    if metric == "":
        column_name = DATASET_OUTPUT_VARIABLES[index]
    else:
        column_name = f"{metric}_{DATASET_OUTPUT_VARIABLES[index]}"
    if column_name not in averaged_df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return averaged_df[column_name].values


def main(path_dataset):
    path_to_dir = Path(path_dataset)
    dataset_folder = dataset_folder_name(path_dataset)

    # Avoid nested/leftover runs when this script is executed repeatedly in notebooks.
    if wandb.run is not None:
        wandb.finish()

    wandb_run = wandb.init(
        mode="online",
        project="Analysis-Plots",
        entity="benthicAI",
        group=dataset_folder,
        name=f"plots_{dataset_folder}",
        reinit=True,
        dir=str(Path(__file__).resolve().parents[1] / "experiments" / "wandb"),
        config={"path_dataset": str(path_dataset), "dataset_folder": dataset_folder},
    )

    coords = load_files(path_to_dir, "coords")
    y_pred = load_files(path_to_dir, "y_pred_mean")
    y_true = load_files(path_to_dir, "y_true")
    y_var = load_files(path_to_dir, "y_pred_var")
    rmse = load_files(path_to_dir, "rmse")
    bias = load_files(path_to_dir, "bias")
    r2 = load_files(path_to_dir, "r2")
    standardized_rmse = load_files(path_to_dir, "rmse", standardized=True)

    metrics_per_station = {
        "y_pred_mean": y_pred,
        "y_pred_var": y_var,
        "rmse": rmse,
        "bias": bias,
        "standardized_rmse": standardized_rmse,
    }
    
    stacked_df = create_stacked_dataframe(coords, y_true, metrics_per_station)

    averaged_df = average_stacked_dataframe_by_coords(stacked_df)

    # Build an averaged dataframe dedicated to r2 (coords + one col per variable)
    averaged_r2_df = create_averaged_r2_dataframe(coords, r2)

    print(f"stacked_df rows={len(stacked_df)}, averaged_df rows={len(averaged_df)}")

    lat = averaged_df["lat"].values
    lon = averaged_df["lon"].values

    metric = "y_pred_mean"
    table_errors = produce_metrics_table(
        averaged_df,
        [metric, "r2", "rmse", "bias", "standardized_rmse"],
        extra_df=averaged_r2_df,
    )
    display(table_errors)
    plot_logger = WandbPlotLogger(wandb_run, namespace=dataset_folder, use_tables=True)
    table_for_log = table_errors.reset_index().rename(columns={"index": "metric"})
    table_key = f"{dataset_folder}_table_errors"
    plot_logger.log_once(table_key, wandb.Table(dataframe=table_for_log))

    for metric in metrics_per_station.keys():
        for index, output_var_name in enumerate(DATASET_OUTPUT_VARIABLES):

            true_values = select_column(averaged_df, "", index)
            pred_values = select_column(averaged_df, metric, index)

            if metric in ["rmse", "bias"]:
                fig_error = plot_error_per_station(
                    pred_values,
                    title=f"{metric} for {output_var_name}",
                    ylabel=f"{metric}",
                )
                
                plot_logger.log(plot_type="error", 
                                variable=output_var_name,
                                metric=metric,
                                fig=fig_error,
                                )
            
            if metric == "y_pred_mean":
                fig_distribution = plot_data_distribution(
                    true_values,
                    pred_values,
                    title=f"Distribution of y_true vs y_pred for {output_var_name}",
                    xlabel=output_var_name,
                    ylabel="Density",
                )
                plot_logger.log(plot_type="distribution", 
                                variable=output_var_name,
                                metric=metric,
                                fig=fig_distribution,
                                )

                fig_correlation = plot_data_correlation(
                    true_values,
                    pred_values,
                    title=f"Correlation of y_true vs y_pred for {output_var_name}",
                    xlabel="y_true",
                    ylabel="y_pred",
                )
                plot_logger.log(plot_type="correlation", 
                                variable=output_var_name,
                                metric=metric,
                                fig=fig_correlation,
                                )

                bathy_path = Path(PATH_MODEL_BATHYMETRY)
                vmin = np.nanmin(pred_values)
                vmax = np.nanmax(pred_values)
                fig_map = plot_maps(
                    lat,
                    lon,
                    pred_values,
                    f"{metric}_" + output_var_name,
                    bathy_nc_path=bathy_path if bathy_path.exists() else None,
                    vmin=vmin,
                    vmax=vmax
                )
                plot_logger.log(plot_type="map_values", 
                                variable=output_var_name,
                                metric=metric,
                                fig=fig_map,
                                )
            
            if metric == "bias":
                bathy_path = Path(PATH_MODEL_BATHYMETRY)
                vmin = np.nanmin(pred_values)
                vmax = np.nanmax(pred_values)
                fig_map = plot_maps(
                    lat,
                    lon,
                    pred_values,
                    f"{metric}_" + output_var_name,
                    bathy_nc_path=bathy_path if bathy_path.exists() else None,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="coolwarm",
                )
                plot_logger.log(plot_type="map_bias", 
                                variable=output_var_name,
                                metric=metric,
                                fig=fig_map,
                                )
    plot_logger.flush()
    wandb.finish()


if __name__ == "__main__":
    default_path_dataset = "/gpfs/projects/acad/bsmfc/benthicAI/models/nn/losveaye"
    parser = argparse.ArgumentParser(description="Generate and log diagnostic plots to Weights & Biases.")
    parser.add_argument(
        "--path-dataset",
        dest="path_dataset",
        default=default_path_dataset,
        help="Path to the dataset/results folder containing metric .npy files.",
    )
    args = parser.parse_args()
    main(args.path_dataset)




