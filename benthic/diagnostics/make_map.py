#! /usr/bin/env python

import mast.figure as mf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import xarray as xr

from mast.ext.database import read_csv_file

if __name__ == "__main__":
    # This part should be edited manually
    bathy_contours = [25, 50, 100, 150]
    stations_style = {"edgecolor": "#DDDDDD"}
    bathy_style = {"colors": "#FFFFFF"}

    if len(sys.argv) != 8:
        print("Usage: make_map.py [1] [2] [3] [4] [5] [6] [7]")
        print("[1] = path to NetCDF file with a map")
        print("[2] = path to a NetCDF file providing bathymetry (2-D)")
        print("[3] = path to .csv file (stations positions)")
        print("[4] = path to .csv file (real values for each station)")
        print("[5] = variable to extract from [1]")
        print("[6] = label for the color bar")
        print("[7] = output file name")
        sys.exit()

    map_file_path = sys.argv[1]
    bathy_file_path = sys.argv[2]
    csv_stations_path = sys.argv[3]
    csv_values_path = sys.argv[4]
    var_name = sys.argv[5]
    color_bar_name = sys.argv[6]
    out_fig_name = sys.argv[7]

    all_paths = [map_file_path, bathy_file_path, csv_stations_path, csv_values_path]
    for path in all_paths:
        if not os.path.isfile(path):
            print("Error: " + path + " does not exist.")
            sys.exit()

    map_ds = xr.open_dataset(map_file_path)
    bathy_ds = xr.open_dataset(bathy_file_path)

    if var_name not in map_ds:
        print("Fatal error: variable '" + var_name + "' not found in " + map_file_path + " !")
        sys.exit()

    # Masking of trait map
    map_ds[var_name].values = np.where(
        map_ds[var_name].values <= 0, np.nan, map_ds[var_name].values
    )

    # Processing .csv inputs
    values_vars = []
    lines_stations = read_csv_file(csv_stations_path, csv_separator=";", decimals_separator=",")
    lines_values = read_csv_file(
        csv_values_path, csv_separator=";", decimals_separator=",", headers=values_vars
    )

    stations = dict()
    for i in range(0, len(lines_stations)):
        stations[lines_stations[i][0]] = (lines_stations[i][2], lines_stations[i][1])

    val_offset = -1
    for i in range(0, len(values_vars)):
        if values_vars[i] == var_name:
            val_offset = i
            break

    if val_offset == -1:
        print("Fatal error: variable '" + var_name + "' not found in " + csv_values_path + " !")
        sys.exit()

    values = dict()
    for i in range(0, len(lines_values)):
        values[lines_values[i][0]] = lines_values[i][val_offset]

    dot_lat, dot_lon, dot_val = [], [], []
    for label in values:
        if label in stations:
            dot_lat.append(stations[label][0])
            dot_lon.append(stations[label][1])
            dot_val.append(values[label])

    fig1 = plt.figure(figsize=(16, 9))
    bs_map = mf.Map(fig1, mf.KNOWN_REGIONS["Black Sea"], rivers_style={"edgecolor": "#1874CD"})
    bs_map.paint_color_mesh(
        map_ds["lat"].values,
        map_ds["lon"].values,
        map_ds[var_name].values,
        values_mask=~np.isnan(map_ds[var_name].values),
        mesh_cmap=cm.viridis,
    )
    bs_map.scatter_with_colors(
        dot_lat, dot_lon, dot_val, dot_style=stations_style, dot_cmap=cm.viridis
    )
    bs_map.add_contours(
        map_ds["lat"].values,
        map_ds["lon"].values,
        bathy_ds["bathymetry"].values,
        bathy_contours,
        values_mask=~np.isnan(map_ds[var_name].values),
        contour_style=bathy_style,
    )
    bs_map.draw_coordinates(show_grid=False)
    bs_map.add_color_bar(color_bar_name)

    plt.savefig(out_fig_name)
    plt.clf()

    # Zoom in on hard-coded coordinates ranges
    zoom_in = [26.5, 32, 41.95, 47]

    lon_start, lon_end = 0, map_ds["lon"].shape[1] - 1
    lat_start, lat_end = 0, map_ds["lat"].shape[0] - 1

    # Longitude range for zoom in: 26.5 °E -> 33 °E
    while map_ds["lon"].values[0, lon_start] < zoom_in[0]:
        lon_start += 1
    while map_ds["lon"].values[0, lon_end] > zoom_in[1]:
        lon_end -= 1

    # Latitude range for zoom in: 41.5 °N -> 47 °N
    while map_ds["lat"].values[lat_start, 0] < zoom_in[2]:
        lat_start += 1
    while map_ds["lat"].values[lat_end, 0] > zoom_in[3]:
        lat_end -= 1

    # Re-doing the figure
    fig2 = plt.figure(figsize=(10, 10))
    bs_crop = mf.Map(fig2, zoom_in, rivers_style={"edgecolor": "#1874CD"})
    bs_crop.paint_color_mesh(
        map_ds["lat"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        map_ds["lon"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        map_ds[var_name].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        values_mask=~np.isnan(
            map_ds[var_name].values[lat_start : lat_end + 1, lon_start : lon_end + 1]
        ),
        mesh_cmap=cm.viridis,
    )
    bs_crop.scatter_with_colors(
        dot_lat, dot_lon, dot_val, dot_style=stations_style, dot_cmap=cm.viridis
    )
    bs_crop.add_contours(
        map_ds["lat"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        map_ds["lon"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        bathy_ds["bathymetry"].values[lat_start : lat_end + 1, lon_start : lon_end + 1],
        bathy_contours,
        contour_style=bathy_style,
    )
    bs_crop.draw_coordinates(show_grid=False)
    bs_crop.add_color_bar(color_bar_name)

    plt.savefig("zoom_" + out_fig_name)
    plt.clf()
