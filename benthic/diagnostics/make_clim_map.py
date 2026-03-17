#! /usr/bin/env python

import mast.figure as mf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import xarray as xr

if __name__ == "__main__":
    # This part should be edited manually
    bathy_contours = [20, 40, 200]
    stations_style = {"edgecolor": "#DDDDDD"}
    bathy_style = {"colors": "#FFFFFF"}

    if len(sys.argv) != 6:
        print("Usage: make_clim_map.py [1] [2] [3] [4] [5]")
        print("[1] = path to NetCDF file with climatological data")
        print("[2] = path to a NetCDF file providing bathymetry (2-D)")
        print("[3] = variable from [1] to plot")
        print("[4] = label for the color bar")
        print("[5] = output file name")
        sys.exit()

    map_file_path = sys.argv[1]
    bathy_file_path = sys.argv[2]
    var_name = sys.argv[3]
    color_bar_name = sys.argv[4]
    out_fig_name = sys.argv[5]

    all_paths = [map_file_path, bathy_file_path]
    for path in all_paths:
        if not os.path.isfile(path):
            print("Error: " + path + " does not exist.")
            sys.exit()

    clim_ds = xr.open_dataset(map_file_path)
    bathy_ds = xr.open_dataset(bathy_file_path)

    if var_name not in clim_ds:
        print("Fatal error: variable '" + var_name + "' not found in " + map_file_path + " !")
        sys.exit()

    # Masking of climatology
    clim_ds[var_name].values = np.where(
        clim_ds[var_name].values <= 0, np.nan, clim_ds[var_name].values
    )

    clim_mean = np.nanmean(clim_ds[var_name].values, axis=0)

    fig1 = plt.figure(figsize=(16, 9))
    bs_map = mf.Map(fig1, mf.KNOWN_REGIONS["Black Sea"], rivers_style={"edgecolor": "#1874CD"})
    bs_map.paint_color_mesh(
        clim_ds["lat"].values,
        clim_ds["lon"].values,
        clim_mean,
        values_mask=~np.isnan(clim_mean),
        mesh_cmap=cm.viridis,
    )
    bs_map.add_contours(
        clim_ds["lat"].values,
        clim_ds["lon"].values,
        bathy_ds["bathymetry"].values,
        bathy_contours,
        values_mask=~np.isnan(clim_mean),
        contour_style=bathy_style,
    )
    bs_map.draw_coordinates(show_grid=False)
    bs_map.add_color_bar(color_bar_name)

    plt.savefig(out_fig_name)
    plt.clf()

    # Zoom in on hard-coded coordinates ranges
    zoom_in = [26.5, 32, 41.95, 47]

    lon_start, lon_end = 0, clim_ds["lon"].shape[1] - 1
    lat_start, lat_end = 0, clim_ds["lat"].shape[0] - 1

    # Longitude range for zoom in: 26.5 °E -> 33 °E
    while clim_ds["lon"].values[0, lon_start] < zoom_in[0]:
        lon_start += 1
    while clim_ds["lon"].values[0, lon_end] > zoom_in[1]:
        lon_end -= 1

    # Latitude range for zoom in: 41.5 °N -> 47 °N
    while clim_ds["lat"].values[lat_start, 0] < zoom_in[2]:
        lat_start += 1
    while clim_ds["lat"].values[lat_end, 0] > zoom_in[3]:
        lat_end -= 1

    # Re-doing the figure
    fig2 = plt.figure(figsize=(10, 10))
    bs_crop = mf.Map(fig2, zoom_in, rivers_style={"edgecolor": "#1874CD"})
    bs_crop.paint_color_mesh(
        clim_ds["lat"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        clim_ds["lon"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        clim_mean[lat_start : lat_end + 1, lon_start : lon_end + 1],
        values_mask=~np.isnan(clim_mean[lat_start : lat_end + 1, lon_start : lon_end + 1]),
        mesh_cmap=cm.viridis,
    )
    bs_crop.add_contours(
        clim_ds["lat"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        clim_ds["lon"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        bathy_ds["bathymetry"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        bathy_contours,
        contour_style=bathy_style,
    )
    bs_crop.draw_coordinates(show_grid=False)
    bs_crop.add_color_bar(color_bar_name)

    plt.savefig("zoom_" + out_fig_name)
    plt.clf()
