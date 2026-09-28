"""
train_bma.py

TRAINING-ONLY pipeline (bottom half of the architecture diagram):

    IMD Gridded Observation -> ground-truth values
                             -> BMA parameter estimation

Fits BMAFuser parameters (weight/bias/sigma per variable) from a
historical archive of paired (IFS forecast, AIFS forecast, IMD
observation) triples, then saves them to bma_params.json for the
operational pipeline (run_forecast_pipeline.py) to load.

IMD's gridded observations are fetched via the `imdlib` package
(https://imdlib.readthedocs.io), which downloads directly -- no manual
data-request form needed. It only provides 'rain', 'tmax', 'tmin', so
`load_imd_observations()` below currently supports 'tp' (mapped to
'rain'); there's no direct IMD equivalent for instantaneous '2t'.

    pip install imdlib

Usage:
    python train_bma.py --start 2023-06-01 --end 2023-09-30 \
        --fxx 12 --vars 2t tp --locations-csv locations.csv \
        --out bma_params.json

locations.csv must have columns: location_id,name,latitude,longitude
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bma_fuser import BMAFuser
from pull_regrid_ifs import get_ifs_on_common_grid
from pull_regrid_aifs import get_aifs_on_common_grid
from common_grid import extract_at_points
from daily_extremes import get_daily_extreme_on_common_grid, daily_extreme_steps_utc


# ---------------------------------------------------------------------
# IMD gridded observations, via the imdlib package (downloads directly,
# no manual data request needed): https://imdlib.readthedocs.io
#
#   pip install imdlib
#
# imdlib gives three variables: 'rain', 'tmax', 'tmin' -- daily, 0.25 deg
# grid, India-only domain (lat 6.5-38.5N, lon 66.5-100.0E). Missing values
# come back as -999.0, not NaN, so they're masked explicitly below.
#
# IMPORTANT: there is no IMD gridded product for instantaneous 2m temp
# ('2t'). 'tmax'/'tmin' are daily extremes, not the same quantity, so
# they are NOT treated as automatic ground truth for '2t' here -- if you
# want to calibrate a temperature variable, fit BMA against 'tmax'/'tmin'
# forecasts (e.g. daily max/min derived from IFS/AIFS) rather than '2t'.
# ---------------------------------------------------------------------

from functools import lru_cache
import xarray as xr
import imdlib as imd

IMD_DATA_DIR = "./imd_data"
IMD_VARIABLE_MAP = {"tp": "rain", "tmax": "tmax", "tmin": "tmin"}
DAILY_EXTREME_VARS = {"tmax", "tmin"}  # these are derived from '2t' via daily_extremes.py, not pulled directly


@lru_cache(maxsize=8)
def _load_imd_year(imd_variable: str, year: int) -> xr.Dataset:
    """Opens (downloading first if needed) one year of an imdlib variable."""
    try:
        data = imd.open_data(imd_variable, year, year, "yearwise", IMD_DATA_DIR)
    except FileNotFoundError:
        imd.get_data(imd_variable, year, year, fn_format="yearwise", file_dir=IMD_DATA_DIR)
        data = imd.open_data(imd_variable, year, year, "yearwise", IMD_DATA_DIR)
    return data.get_xarray()


def load_imd_observations(valid_time: datetime, variable: str,
                           lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Ground truth for BMA training, from IMD's 0.25 deg gridded daily
    dataset via imdlib. Returns one value per (lat, lon) point, using
    nearest-neighbor lookup (the source grid is already 0.25 deg, the
    same resolution as our common grid).

    Only variables in IMD_VARIABLE_MAP are supported. Points outside
    IMD's India-only domain, or IMD's -999.0 missing-value cells, come
    back as NaN -- BMAFuser.fit_variable already filters NaNs out.
    """
    if variable not in IMD_VARIABLE_MAP:
        raise NotImplementedError(
            f"No IMD ground-truth source configured for variable {variable!r}. "
            f"imdlib only provides {list(IMD_VARIABLE_MAP)} -- add a mapping in "
            "IMD_VARIABLE_MAP if you have a matching forecast variable to calibrate against."
        )
    imd_variable = IMD_VARIABLE_MAP[variable]

    ds = _load_imd_year(imd_variable, valid_time.year)
    da_day = ds[imd_variable].sel(time=np.datetime64(valid_time.date()), method="nearest")

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    in_domain = (lats >= 6.5) & (lats <= 38.5) & (lons >= 66.5) & (lons <= 100.0)

    obs = np.full(len(lats), np.nan)
    if in_domain.any():
        lat_da = xr.DataArray(lats[in_domain], dims="points")
        lon_da = xr.DataArray(lons[in_domain], dims="points")
        vals = da_day.sel(lat=lat_da, lon=lon_da, method="nearest").values
        obs[in_domain] = np.where(vals <= -999.0, np.nan, vals)

    return obs


def build_training_set(init_times: list[datetime], variables: list[str],
                        lats: np.ndarray, lons: np.ndarray,
                        fxx: int | None = None, day_offset: int = 1,
                        step_hours: int = 3) -> dict[str, dict]:
    """
    For each historical init_time, pulls IFS+AIFS and pairs them with the
    matching IMD observation:

      - "tmax"/"tmin": derived by aggregating '2t' across every forecast
        step in the target day (init_time + day_offset), one call to
        get_daily_extreme_on_common_grid per model, per init_time.
      - anything else (e.g. "tp"): pulled directly at forecast hour `fxx`,
        as before.

    Returns {variable: {"obs": np.ndarray, "ifs": np.ndarray, "aifs": np.ndarray}}
    (all points from all init_times concatenated together).
    """
    training = {v: {"obs": [], "ifs": [], "aifs": []} for v in variables}
    instant_vars = [v for v in variables if v not in DAILY_EXTREME_VARS]
    extreme_vars = [v for v in variables if v in DAILY_EXTREME_VARS]

    if instant_vars and fxx is None:
        raise ValueError(f"--fxx is required to train on {instant_vars}")

    for init_time in init_times:
        init_str = init_time.strftime("%Y-%m-%d %H:%M")

        # --- instantaneous variables (existing single-fxx path) ---
        if instant_vars:
            valid_time = init_time + timedelta(hours=fxx)
            try:
                ds_ifs, _, _ = get_ifs_on_common_grid(init_str, fxx=fxx, variables=[f":{v}:" for v in instant_vars])
                ds_aifs, _, _ = get_aifs_on_common_grid(init_str, fxx=fxx, variables=[f":{v}:" for v in instant_vars])
                pts_ifs = extract_at_points(ds_ifs, lats, lons)
                pts_aifs = extract_at_points(ds_aifs, lats, lons)
                for var in instant_vars:
                    if var not in pts_ifs or var not in pts_aifs:
                        continue
                    obs = load_imd_observations(valid_time, var, lats, lons)
                    training[var]["obs"].append(obs)
                    training[var]["ifs"].append(pts_ifs[var])
                    training[var]["aifs"].append(pts_aifs[var])
            except Exception as e:
                print(f"[train] skipping {init_time} (instant vars): {e}")

        # --- daily-extreme variables (tmax/tmin, derived from 2t) ---
        if extreme_vars:
            valid_date = init_time + timedelta(days=day_offset)
            fxx_list = daily_extreme_steps_utc(24 * day_offset, 24 * (day_offset + 1), step_hours)
            for var in extreme_vars:
                mode = "max" if var == "tmax" else "min"
                try:
                    ds_ifs, _, _ = get_daily_extreme_on_common_grid(
                        get_ifs_on_common_grid, init_str, fxx_list, mode=mode)
                    ds_aifs, _, _ = get_daily_extreme_on_common_grid(
                        get_aifs_on_common_grid, init_str, fxx_list, mode=mode)
                except Exception as e:
                    print(f"[train] skipping {init_time} ({var}): {e}")
                    continue
                pts_ifs = extract_at_points(ds_ifs, lats, lons)   # key is '2t' (base_variable)
                pts_aifs = extract_at_points(ds_aifs, lats, lons)
                obs = load_imd_observations(valid_date, var, lats, lons)
                training[var]["obs"].append(obs)
                training[var]["ifs"].append(pts_ifs["2t"])
                training[var]["aifs"].append(pts_aifs["2t"])

    for var in variables:
        for key in ("obs", "ifs", "aifs"):
            training[var][key] = np.concatenate(training[var][key]) if training[var][key] else np.array([])

    return training


def main():
    p = argparse.ArgumentParser(description="Train BMA parameters from historical IFS/AIFS vs IMD obs.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--fxx", type=int, default=None,
                    help="Forecast hour to train on (required for non-tmax/tmin variables, e.g. tp)")
    p.add_argument("--day-offset", type=int, default=1,
                    help="Which forecast day (relative to init) to build tmax/tmin from")
    p.add_argument("--step-hours", type=int, default=3,
                    help="Step spacing (hours) used to sample '2t' when building tmax/tmin")
    p.add_argument("--vars", nargs="+", default=["tmax", "tmin"], help="Variables to fit, e.g. tmax tmin tp")
    p.add_argument("--locations-csv", required=True,
                    help="CSV with columns location_id,name,latitude,longitude used as training points")
    p.add_argument("--out", default="bma_params.json")
    args = p.parse_args()

    locations = pd.read_csv(args.locations_csv)
    lats = locations["latitude"].to_numpy()
    lons = locations["longitude"].to_numpy()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    init_times = [start + timedelta(days=d) for d in range((end - start).days + 1)]

    training = build_training_set(init_times, args.vars, lats, lons,
                                   fxx=args.fxx, day_offset=args.day_offset, step_hours=args.step_hours)

    fuser = BMAFuser()
    for var in args.vars:
        d = training[var]
        if len(d["obs"]) == 0:
            print(f"[train] WARNING: no training data for {var}, skipping")
            continue
        params = fuser.fit_variable(var, d["obs"], d["ifs"], d["aifs"])
        print(f"[train] {var}: w_ifs={params.weight_ifs:.3f} w_aifs={params.weight_aifs:.3f} "
              f"bias_ifs={params.bias_ifs:.3f} bias_aifs={params.bias_aifs:.3f} "
              f"sigma_ifs={params.sigma_ifs:.3f} sigma_aifs={params.sigma_aifs:.3f} n={params.n_train}")

    fuser.save_params(args.out)
    print(f"[train] Saved fitted BMA parameters -> {args.out}")


if __name__ == "__main__":
    main()