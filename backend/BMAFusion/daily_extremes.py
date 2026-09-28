"""
daily_extremes.py

Neither IFS nor AIFS output daily tmax/tmin directly -- both only give
instantaneous 2m temperature ('2t') at each forecast step. IMD's ground
truth for temperature (via imdlib) is 'tmax'/'tmin', which are inherently
daily quantities. To calibrate BMA on a like-for-like target (Option A),
we build tmax/tmin ourselves: pull '2t' across every forecast step that
falls within the target day, then take the max/min across those steps.

Generic over the puller (get_ifs_on_common_grid or get_aifs_on_common_grid)
so IFS and AIFS use identical aggregation logic -- important, since any
asymmetry here would bias the BMA fit before it even sees the data.
"""

from __future__ import annotations

from typing import Callable

import xarray as xr


def daily_extreme_steps_utc(hour_start: int, hour_end: int, step_hours: int = 3) -> list[int]:
    """
    Forecast-hour list spanning one UTC day at a fixed step, e.g.
    daily_extreme_steps_utc(24, 48, 3) -> [24, 27, ..., 48] for the day
    after a 00Z init. Override step_hours / call with your own list if
    your model's actual output step schedule differs (IFS open-data step
    spacing coarsens at longer lead times).
    """
    return list(range(hour_start, hour_end + 1, step_hours))


def get_daily_extreme_on_common_grid(puller: Callable, init_time: str, fxx_list: list[int],
                                      mode: str = "max", base_variable: str = "2t",
                                      resolution: float = 0.25):
    """
    Parameters
    ----------
    puller : get_ifs_on_common_grid or get_aifs_on_common_grid
    init_time : model init time, e.g. "2024-03-01 00:00"
    fxx_list : forecast hours to aggregate over -- must span the target day
               (see daily_extreme_steps_utc)
    mode : "max" or "min"
    base_variable : the instantaneous field to aggregate (default '2t')

    Returns
    -------
    ds_extreme : xr.Dataset, one variable (named base_variable) holding
                 the max/min across all fxx_list steps, on the common grid
    lat, lon : common grid coordinates
    """
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")
    if not fxx_list:
        raise ValueError("fxx_list must span at least one forecast step")

    per_step = []
    lat = lon = None
    for fxx in fxx_list:
        ds, lat, lon = puller(init_time, fxx=fxx, variables=[f":{base_variable}:"], resolution=resolution)
        per_step.append(ds.expand_dims(step=[fxx]))

    stacked = xr.concat(per_step, dim="step")
    ds_extreme = stacked.max(dim="step") if mode == "max" else stacked.min(dim="step")
    return ds_extreme, lat, lon