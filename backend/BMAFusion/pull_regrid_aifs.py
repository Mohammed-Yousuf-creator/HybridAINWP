"""
pull_regrid_aifs.py

Pulls ECMWF AIFS (ML-based) GRIB2 fields via Herbie and regrids them
onto the same common regular lat/lon grid used by pull_regrid_ifs.py,
for use in a BMA (Bayesian Model Averaging) forecast fuser.

Grid definition and regridding now live in common_grid.py, shared with
pull_regrid_ifs.py, so both scripts always land on identical points.

Usage (CLI):
    python pull_regrid_aifs.py --date "2024-03-01 00:00" --fxx 12 \
        --vars 2t 10u 10v tp --out aifs_common.nc

Usage (as a library):
    from pull_regrid_aifs import get_aifs_on_common_grid
    ds, lat, lon = get_aifs_on_common_grid("2024-03-01 00:00", fxx=12,
                                            variables=[":2t:", ":10u:", ":10v:"])

Note: AIFS model naming has shifted a bit across Herbie versions
(e.g. "aifs", "aifs-single"). If Herbie() raises a model-not-found
error, check `Herbie("...", model="aifs").PRODUCTS` or your installed
Herbie's model source list and adjust MODEL_NAME below.
"""

import argparse
import xarray as xr
from herbie import Herbie

from common_grid import define_common_grid, regrid_to_common

MODEL_NAME = "aifs"  # adjust if your Herbie version uses e.g. "aifs-single"


# ---------------------------------------------------------------------
# Main pull + regrid routine
# ---------------------------------------------------------------------
def get_aifs_on_common_grid(date: str,
                             fxx: int = 12,
                             variables=(":2t:", ":10u:", ":10v:", ":tp:"),
                             resolution: float = 0.25):
    """
    Downloads the requested AIFS fields for one init time / forecast hour,
    merges them, and regrids to the common grid.

    Parameters
    ----------
    date : str  - init time, e.g. "2024-03-01 00:00"
    fxx  : int  - forecast hour (e.g. 12 for F12)
    variables : iterable of Herbie regex search strings (see cheat sheet)
    resolution : degrees, passed to define_common_grid

    Returns
    -------
    ds_common : xr.Dataset on the common grid
    lat, lon  : np.ndarray, the common grid coordinates used
    """
    H = Herbie(date, model=MODEL_NAME, product="oper", fxx=fxx)

    datasets = []
    for var in variables:
        try:
            datasets.append(H.xarray(var))
        except Exception as e:
            print(f"[aifs] WARNING: could not extract {var!r}: {e}")

    if not datasets:
        raise RuntimeError("No AIFS variables were successfully extracted.")

    ds = xr.merge(datasets, compat="override")

    lat, lon = define_common_grid(resolution=resolution)
    ds_common = regrid_to_common(ds, lat, lon)

    ds_common.attrs["source_model"] = MODEL_NAME
    ds_common.attrs["init_time"] = str(date)
    ds_common.attrs["fxx"] = fxx

    return ds_common, lat, lon


def main():
    p = argparse.ArgumentParser(description="Pull and regrid AIFS data for BMA fusion.")
    p.add_argument("--date", required=True, help='Init time, e.g. "2024-03-01 00:00"')
    p.add_argument("--fxx", type=int, default=12, help="Forecast hour")
    p.add_argument("--vars", nargs="+", default=["2t", "10u", "10v", "tp"],
                    help="Variable short names (without colons), e.g. 2t 10u 10v tp")
    p.add_argument("--resolution", type=float, default=0.25, help="Common grid resolution (deg)")
    p.add_argument("--out", default="aifs_common.nc", help="Output NetCDF path")
    args = p.parse_args()

    search_strings = [f":{v}:" for v in args.vars]
    ds_common, lat, lon = get_aifs_on_common_grid(
        args.date, fxx=args.fxx, variables=search_strings, resolution=args.resolution
    )

    ds_common.to_netcdf(args.out)
    print(f"[aifs] Saved regridded dataset -> {args.out}")
    print(f"[aifs] Common grid: lat {lat[0]}..{lat[-1]} ({len(lat)} pts), "
          f"lon {lon[0]}..{lon[-1]} ({len(lon)} pts)")


if __name__ == "__main__":
    main()