"""
run_forecast_pipeline.py

Operational pipeline -- the top half of the architecture diagram, from
"pull IFS/AIFS" through "probabilistic forecast", stopping right before
MySQL/FastAPI/React:

    IFS  ---\
              >-- regrid to common grid --> BMA fuse --> probabilistic forecast
    AIFS ---/

This does NOT touch a database, an API framework, or a frontend -- it
returns plain Python dicts/lists shaped EXACTLY like the tables you
specified, plus a ready-to-render map array:

    "locations"     -> matches your `locations` table
    "forecast_run"  -> matches your `forecast_runs` table
    "forecasts"     -> matches your `forecasts` table
    "map_features"  -> GeoJSON FeatureCollection for Leaflet/Mapbox/deck.gl

Wire the returned dicts into your own DB-insert code and API response --
that part is intentionally left out.

Two kinds of variables are handled, matching train_bma.py:
  - instantaneous (e.g. "tp", "10u", "10v"): pulled directly at each fxx
    in `fxx_list`.
  - daily-extreme ("tmax"/"tmin"): derived from '2t' by aggregating every
    forecast step within the target day (see daily_extremes.py) -- one
    entry per day_offset in `day_offsets`, NOT per fxx.

Usage (as a library):
    from run_forecast_pipeline import run_pipeline
    result = run_pipeline(
        init_time="2024-03-01 00:00",
        fxx_list=[12, 24],           # for instantaneous variables
        variables=["tp", "tmax", "tmin"],
        locations=[{"location_id": "delhi", "name": "New Delhi",
                    "latitude": 28.6139, "longitude": 77.2090}],
        bma_params_path="bma_params.json",
        model_version="bma-v1",
        day_offsets=[1, 2],          # for tmax/tmin: day 1 and day 2 ahead
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from common_grid import extract_at_points
from pull_regrid_ifs import get_ifs_on_common_grid
from pull_regrid_aifs import get_aifs_on_common_grid
from daily_extremes import get_daily_extreme_on_common_grid, daily_extreme_steps_utc
from bma_fuser import BMAFuser
from geojson_utils import build_map_feature

DAILY_EXTREME_VARS = {"tmax", "tmin"}

# Raw GRIB units -> display units + conversion function applied ONLY to
# the final probabilistic output (BMA itself is fit/predicted in raw units).
#
# tmax/tmin are identity here: BMAFuser was fit against IMD's tmax/tmin
# (already in Celsius) as the "obs" target, so the fitted bias terms
# already absorb the Kelvin->Celsius offset -- predict() output for
# these variables comes out in Celsius already. Do NOT subtract 273.15
# again here, that would double-convert.
VARIABLE_UNITS = {
    "2t":   {"unit": "°C",  "convert": lambda k: k - 273.15},   # Kelvin -> Celsius
    "tmax": {"unit": "°C",  "convert": lambda c: c},            # already Celsius (via BMA bias)
    "tmin": {"unit": "°C",  "convert": lambda c: c},            # already Celsius (via BMA bias)
    "tp":   {"unit": "mm",  "convert": lambda m: m * 1000.0},   # meters -> millimeters
    "10u":  {"unit": "m/s", "convert": lambda v: v},
    "10v":  {"unit": "m/s", "convert": lambda v: v},
}

# Optional event-exceedance thresholds, given in the SAME space BMA
# samples in for that variable (raw GRIB units for instantaneous
# variables; Celsius for tmax/tmin, per the note above).
EVENT_THRESHOLDS = {
    "tp": 0.001,   # 0.001 m = 1 mm rain
}


def _convert_forecast(fc: dict, convert) -> dict:
    return {
        k: (convert(v) if isinstance(v, float) and k != "probability" else v)
        for k, v in fc.items()
    }


def _build_records(run_id, valid_time_iso, var, ifs_vals, aifs_vals, locations, fuser):
    """Shared per-variable record + map-feature builder for both branches below."""
    unit_cfg = VARIABLE_UNITS.get(var, {"unit": "", "convert": lambda x: x})
    threshold = EVENT_THRESHOLDS.get(var)

    records, features = [], []
    for i, loc in enumerate(locations):
        fc_raw = fuser.predict(var, float(ifs_vals[i]), float(aifs_vals[i]), event_threshold=threshold)
        fc = _convert_forecast(fc_raw, unit_cfg["convert"])

        record = {
            "forecast_id": str(uuid.uuid4()),
            "run_id": run_id,
            "location_id": loc["location_id"],
            "valid_time": valid_time_iso,
            "variable": var,
            "unit": unit_cfg["unit"],
            "central_value": fc["central_value"],
            "p10": fc["p10"], "p25": fc["p25"], "p50": fc["p50"],
            "p75": fc["p75"], "p90": fc["p90"],
            "probability": fc["probability"],
        }
        records.append(record)
        features.append(build_map_feature(loc, record))
    return records, features


def run_pipeline(init_time: str, fxx_list: list[int], variables: list[str],
                  locations: list[dict], bma_params_path: str, model_version: str,
                  day_offsets: Optional[list[int]] = None, step_hours: int = 3) -> dict:
    """
    Parameters
    ----------
    init_time : str            e.g. "2024-03-01 00:00" -- forecast_reference_time
    fxx_list  : list[int]      forecast hours for INSTANTANEOUS variables, e.g. [12, 24, 36]
    variables : list[str]      e.g. ["tp", "tmax", "tmin", "10u"]
    locations : list[dict]     each {"location_id", "name", "latitude", "longitude"} --
                                exactly your `locations` table rows
    bma_params_path : str      path to bma_params.json produced by train_bma.py
    model_version : str        version string recorded in forecast_runs
    day_offsets : list[int]    which forecast days (relative to init) to produce tmax/tmin
                                for -- required if "tmax"/"tmin" are in `variables`
    step_hours : int           step spacing used to sample '2t' when building tmax/tmin

    Returns
    -------
    {
      "locations":     [...],   # pass-through, matches `locations` table
      "forecast_run":  {...},   # matches `forecast_runs` table
      "forecasts":     [...],   # matches `forecasts` table
      "map_features":  {...}    # GeoJSON FeatureCollection for the frontend map
    }
    """
    fuser = BMAFuser()
    fuser.load_params(bma_params_path)

    lats = np.array([loc["latitude"] for loc in locations])
    lons = np.array([loc["longitude"] for loc in locations])

    instant_vars = [v for v in variables if v not in DAILY_EXTREME_VARS]
    extreme_vars = [v for v in variables if v in DAILY_EXTREME_VARS]
    if extreme_vars and not day_offsets:
        raise ValueError(f"day_offsets is required to produce {extreme_vars}")

    run_id = str(uuid.uuid4())
    forecast_run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "forecast_reference_time": init_time,
        "model_version": model_version,
    }

    forecasts_records: list[dict] = []
    map_features: list[dict] = []
    init_dt = datetime.strptime(init_time, "%Y-%m-%d %H:%M")

    # --- instantaneous variables, pulled per fxx ---
    if instant_vars:
        search_strings = [f":{v}:" for v in instant_vars]
        for fxx in fxx_list:
            valid_time_iso = (init_dt + timedelta(hours=fxx)).isoformat()
            ds_ifs, _, _ = get_ifs_on_common_grid(init_time, fxx=fxx, variables=search_strings)
            ds_aifs, _, _ = get_aifs_on_common_grid(init_time, fxx=fxx, variables=search_strings)
            pts_ifs = extract_at_points(ds_ifs, lats, lons)
            pts_aifs = extract_at_points(ds_aifs, lats, lons)

            for var in instant_vars:
                if var not in pts_ifs or var not in pts_aifs:
                    print(f"[pipeline] {var} missing from IFS or AIFS output at fxx={fxx}, skipping")
                    continue
                records, features = _build_records(
                    run_id, valid_time_iso, var, pts_ifs[var], pts_aifs[var], locations, fuser)
                forecasts_records.extend(records)
                map_features.extend(features)

    # --- daily-extreme variables (tmax/tmin), one entry per day_offset ---
    for day_offset in (day_offsets or []):
        valid_date_iso = (init_dt + timedelta(days=day_offset)).date().isoformat()
        fxx_list_day = daily_extreme_steps_utc(24 * day_offset, 24 * (day_offset + 1), step_hours)

        for var in extreme_vars:
            mode = "max" if var == "tmax" else "min"
            ds_ifs, _, _ = get_daily_extreme_on_common_grid(get_ifs_on_common_grid, init_time, fxx_list_day, mode=mode)
            ds_aifs, _, _ = get_daily_extreme_on_common_grid(get_aifs_on_common_grid, init_time, fxx_list_day, mode=mode)
            pts_ifs = extract_at_points(ds_ifs, lats, lons)   # key is '2t' (the base_variable)
            pts_aifs = extract_at_points(ds_aifs, lats, lons)

            records, features = _build_records(
                run_id, valid_date_iso, var, pts_ifs["2t"], pts_aifs["2t"], locations, fuser)
            forecasts_records.extend(records)
            map_features.extend(features)

    return {
        "locations": locations,
        "forecast_run": forecast_run,
        "forecasts": forecasts_records,
        "map_features": {"type": "FeatureCollection", "features": map_features},
    }


if __name__ == "__main__":
    # Minimal example wiring -- replace with your real locations table rows.
    example_locations = [
        {"location_id": "delhi", "name": "New Delhi", "latitude": 28.6139, "longitude": 77.2090},
        {"location_id": "mumbai", "name": "Mumbai", "latitude": 19.0760, "longitude": 72.8777},
    ]
    result = run_pipeline(
        init_time="2024-03-01 00:00",
        fxx_list=[12, 24],
        variables=["tp", "tmax", "tmin"],
        locations=example_locations,
        bma_params_path="bma_params.json",
        model_version="bma-v1",
        day_offsets=[1],
    )
    import json
    print(json.dumps(result, indent=2, default=str))