"""
common_grid.py

Shared grid definition, coordinate standardization, and regridding /
point-extraction helpers used by pull_regrid_ifs.py, pull_regrid_aifs.py,
train_bma.py, and run_forecast_pipeline.py.

Centralizing this here (rather than duplicating it in every script)
guarantees IFS and AIFS data always land on identical grid points before
being handed to the BMA fuser.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def define_common_grid(resolution: float = 0.25,
                        lat_range: tuple = (90.0, -90.0),
                        lon_range: tuple = (0.0, 359.75)) -> tuple[np.ndarray, np.ndarray]:
    """
    Defines the shared target grid both IFS and AIFS data get interpolated
    onto before being handed to the BMA fuser.

    Returns
    -------
    lat : np.ndarray, descending (90 -> -90)
    lon : np.ndarray, ascending (0 -> 359.75), 0-360 convention
    """
    n_lat = round((lat_range[0] - lat_range[1]) / resolution) + 1
    n_lon = round((lon_range[1] - lon_range[0]) / resolution) + 1
    lat = np.linspace(lat_range[0], lat_range[1], n_lat)
    lon = np.linspace(lon_range[0], lon_range[1], n_lon)
    return lat, lon


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Renames lat/lon -> latitude/longitude and forces 0-360 longitude convention."""
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    lon = ds["longitude"]
    if float(lon.min()) < 0:
        ds = ds.assign_coords(longitude=(lon % 360)).sortby("longitude")
    return ds


def regrid_to_common(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Linear interpolation of a full dataset onto the common lat/lon grid."""
    ds = standardize_coords(ds)
    return ds.interp(latitude=lat, longitude=lon, method="linear")


def extract_at_points(ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray) -> dict[str, np.ndarray]:
    """
    Interpolates every data variable in `ds` at a set of point lat/lons
    (one value per point, not a grid) -- used to pull forecast values out
    at specific station/city locations for the BMA fuser and DB records.

    Parameters
    ----------
    ds   : xr.Dataset, already regridded onto (or natively on) a regular
           lat/lon grid with 'latitude'/'longitude' coords.
    lats, lons : array-like, the point locations to sample. Longitudes may
           be given in -180..180 or 0..360; both are handled.

    Returns
    -------
    {variable_name: np.ndarray of length len(lats)}
    """
    ds = standardize_coords(ds)
    lons_0_360 = np.mod(np.asarray(lons, dtype=float), 360)
    lat_da = xr.DataArray(np.asarray(lats, dtype=float), dims="points")
    lon_da = xr.DataArray(lons_0_360, dims="points")
    interpolated = ds.interp(latitude=lat_da, longitude=lon_da, method="linear")
    return {var: interpolated[var].values for var in interpolated.data_vars}