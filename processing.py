import dask
import numpy as np
import xarray as xr

dask.config.set(scheduler="synchronous")
# Load the to_infer dataset
ds_infer = xr.open_zarr("to_infer.zarr")
ds_infer.load()

# Load the mesh mask to get the full grid structure
xm = xr.open_dataset("data/model/mesh_mask.nc")
xb = xr.open_dataset("/gpfs/home/acad/ulg-mast/lvandenb/traits/predictors/clim_2008_2018_.nc")

# Get the full 2D grid coordinates
latitude_2d = xb["lat"].values  # Shape: (nlatitude, nlongitude)
longitude_2d = xb["lon"].values  # Shape: (nlatitude, nlongitude)
mask_2d = xm["tmask"][0, 0, :, :].values  # Shape: (nlatitude, nlongitude)

nlatitude, nlongitude = latitude_2d.shape
time_steps = 365

print(f"Grid shape: {nlatitude} x {nlongitude}")
print(f"Number of ocean points: {np.sum(mask_2d)}")

# Extract variables from X (same structure as Xi in training data)
X = ds_infer["X"].values  # Shape: (n_ocean_points, n_features)

# Extract the 3 static variables and 8 time series
bathymetry_flat = X[:, 0]
gravel_flat = X[:, 1]
mud_sand_ratio_flat = X[:, 2]

# Split the remaining columns into 8 time series variables
(
    votemper_flat,
    vosaline_flat,
    dox_flat,
    par_flat,
    sCSED_flat,
    botfluxPOC_flat,
    botstress_flat,
    fCSED_flat,
) = np.split(X[:, 3:], 8, axis=1)

print(f"Shape of votemper_flat: {votemper_flat.shape}")  # Should be (n_ocean_points, 365)

# Create empty 2D/3D arrays filled with NaN
bathymetry_2d = np.full((nlatitude, nlongitude), np.nan, dtype=np.float32)
gravel_2d = np.full((nlatitude, nlongitude), np.nan, dtype=np.float32)
mud_sand_ratio_2d = np.full((nlatitude, nlongitude), np.nan, dtype=np.float32)

votemper_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
vosaline_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
dox_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
par_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
sCSED_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
botfluxPOC_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
botstress_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)
fCSED_3d = np.full((nlatitude, nlongitude, time_steps), np.nan, dtype=np.float32)

# Fill the arrays at ocean points
ocean_idx = 0
for i in range(nlatitude):
    for j in range(nlongitude):
        if mask_2d[i, j] != 0:  # Ocean point
            bathymetry_2d[i, j] = bathymetry_flat[ocean_idx]
            gravel_2d[i, j] = gravel_flat[ocean_idx]
            mud_sand_ratio_2d[i, j] = mud_sand_ratio_flat[ocean_idx]

            votemper_3d[i, j, :] = votemper_flat[ocean_idx, :]
            vosaline_3d[i, j, :] = vosaline_flat[ocean_idx, :]
            dox_3d[i, j, :] = dox_flat[ocean_idx, :]
            par_3d[i, j, :] = par_flat[ocean_idx, :]
            sCSED_3d[i, j, :] = sCSED_flat[ocean_idx, :]
            botfluxPOC_3d[i, j, :] = botfluxPOC_flat[ocean_idx, :]
            botstress_3d[i, j, :] = botstress_flat[ocean_idx, :]
            fCSED_3d[i, j, :] = fCSED_flat[ocean_idx, :]

            ocean_idx += 1

print(f"Filled {ocean_idx} ocean points")

# Get 1D coordinate arrays (assuming regular grid)
latitude_1d = latitude_2d[:, 0]  # First column
longitude_1d = longitude_2d[0, :]  # First row
time = np.arange(1, 366, dtype=np.int64)

# Create the new xarray Dataset
ds_grid = xr.Dataset(
    data_vars={
        # Static variables (latitude, longitude)
        "bathymetry": (["latitude", "longitude"], bathymetry_2d),
        "gravel": (["latitude", "longitude"], gravel_2d),
        "mud_sand_ratio": (["latitude", "longitude"], mud_sand_ratio_2d),
        # Time series variables (latitude, longitude, time)
        "votemper": (["latitude", "longitude", "time"], votemper_3d),
        "vosaline": (["latitude", "longitude", "time"], vosaline_3d),
        "dox": (["latitude", "longitude", "time"], dox_3d),
        "par": (["latitude", "longitude", "time"], par_3d),
        "sCSED": (["latitude", "longitude", "time"], sCSED_3d),
        "botfluxPOC": (["latitude", "longitude", "time"], botfluxPOC_3d),
        "botstress": (["latitude", "longitude", "time"], botstress_3d),
        "fCSED": (["latitude", "longitude", "time"], fCSED_3d),
    },
    coords={
        "latitude": (["latitude"], latitude_1d.astype(np.float32)),
        "longitude": (["longitude"], longitude_1d.astype(np.float32)),
        "time": (["time"], time),
    },
    attrs={"Description": "Environmental predictors on 2D grid for benthic trait prediction"},
)

print(ds_grid)

# Chunk dataset for performance
dataset = ds_grid.chunk({"time": -1, "longitude": -1, "latitude": -1})
for var in dataset.data_vars:
    if "chunks" in dataset[var].encoding:
        del dataset[var].encoding["chunks"]

dataset.to_zarr("predictors_grid_chunked.zarr", mode="w")

print("\nDataset sauvegardé dans 'predictors_grid.zarr'")
print(f"Dimensions: latitude={nlatitude}, longitude={nlongitude}, time={time_steps}")
