# BMAFusion

BMAFusion combines ECMWF IFS numerical weather prediction forecasts with
ECMWF AIFS machine-learning forecasts using Bayesian Model Averaging (BMA).
It produces calibrated probabilistic forecasts for configured locations and
returns records that can be passed to the backend database/API layer.

The module is intentionally split into two workflows:

1. **Training**: historical IFS and AIFS forecasts are compared with IMD
	 gridded observations to estimate BMA weights, biases, and uncertainties.
2. **Forecasting**: the fitted parameters are applied to new IFS and AIFS
	 forecasts to produce a central value, quantiles, and optional event
	 probabilities.

## Directory Overview

| File | Purpose |
| --- | --- |
| `bma_fuser.py` | Core two-component Gaussian-mixture BMA model. |
| `TrainBMA.py` | Downloads/loads IMD observations, builds training pairs, and fits parameters. |
| `run_forecast_pipeline.py` | Operational IFS/AIFS fusion pipeline and output record builder. |
| `pull_regrid_ifs.py` | Retrieves IFS GRIB fields through Herbie and regrids them. |
| `pull_regrid_aifs.py` | Retrieves AIFS GRIB fields through Herbie and regrids them. |
| `common_grid.py` | Defines the shared 0.25-degree grid and point extraction helpers. |
| `daily_extremes.py` | Builds daily `tmax`/`tmin` from forecast-step `2t` values. |
| `geojson_utils.py` | Builds GeoJSON forecast features and feature collections for map output. |
| `locations.csv` | Example location input with IDs, names, and coordinates. |
| `bma_params.json` | Fitted BMA parameters consumed by the forecast pipeline. |
| `pyproject.toml` | Project metadata and Python version requirement. |

Generated/downloaded files such as GRIB2, NetCDF, plots, and IMD data should
be treated as data artifacts rather than source code.

## Requirements

- Python 3.13 or newer (`pyproject.toml` specifies `>=3.13`)
- Access to the ECMWF data sources supported by the installed Herbie version
- Network access when Herbie or `imdlib` needs to download data
- An environment containing the imported scientific packages:
	`numpy`, `scipy`, `pandas`, `xarray`, `netCDF4`, `herbie-data`, and
	`imdlib`

The current `pyproject.toml` declares project metadata but does not yet list
these runtime dependencies. Install them in the project environment before
running the scripts. The local `.venv/` directory, when present, is an
environment and is not required to be committed or copied as application
source.

From this directory, activate the environment and install the dependencies
using the package manager used by the project. For example:

```bash
source .venv/bin/activate
python -m pip install numpy scipy pandas xarray netCDF4 herbie-data imdlib
```

## Shared Grid and Coordinates

Both models are interpolated onto the same regular grid before fusion:

- Latitude: 90 to -90 degrees, descending
- Longitude: 0 to 359.75 degrees, ascending
- Default resolution: 0.25 degrees

Location longitudes may be supplied in either `-180..180` or `0..360`; the
point extraction helper normalizes them to `0..360`.

The location CSV and location dictionaries use these columns/keys:

```text
location_id,name,latitude,longitude
```

## Training Workflow

`TrainBMA.py` fits one BMA parameter set per requested variable. Each model
component is represented as:

```text
observation ~ w_ifs  * Normal(IFS  + bias_ifs,  sigma_ifs)
					 + w_aifs * Normal(AIFS + bias_aifs, sigma_aifs)
```

The fitted weights sum to one. At least 20 valid paired samples are required
for each variable; missing and non-finite values are filtered before fitting.

IMD currently supplies the training targets used by this module for:

- `tp`, mapped to IMD `rain`
- `tmax`, mapped to IMD `tmax`
- `tmin`, mapped to IMD `tmin`

Daily extremes are derived from `2t` forecast steps over the target day, so
they are compared with like-for-like daily IMD observations. Instantaneous
`2t` does not have a direct IMD training target in the current implementation.

Example for precipitation:

```bash
python TrainBMA.py \
	--start 2023-06-01 \
	--end 2023-09-30 \
	--fxx 12 \
	--vars tp \
	--locations-csv locations.csv \
	--out bma_params.json
```

Example for daily temperature extremes:

```bash
python TrainBMA.py \
	--start 2023-06-01 \
	--end 2023-09-30 \
	--vars tmax tmin \
	--day-offset 1 \
	--step-hours 3 \
	--locations-csv locations.csv \
	--out bma_params.json
```

The command downloads IMD data into `./imd_data` when it is not already
available and writes a JSON parameter file. Keep the parameter file matched to
the variables used by the operational pipeline.

## Forecast Workflow

The main library entry point is `run_pipeline`:

```python
from run_forecast_pipeline import run_pipeline

locations = [
		{
				"location_id": "delhi",
				"name": "New Delhi",
				"latitude": 28.6139,
				"longitude": 77.2090,
		}
]

result = run_pipeline(
		init_time="2024-03-01 00:00",
		fxx_list=[12, 24, 36],
		variables=["tp", "tmax", "tmin", "10u", "10v"],
		locations=locations,
		bma_params_path="bma_params.json",
		model_version="bma-v1",
		day_offsets=[1, 2],
)
```

Instantaneous variables are evaluated at every value in `fxx_list`. `tmax`
and `tmin` are calculated from all `2t` steps in each requested
`day_offsets` day and therefore do not use `fxx_list`. `day_offsets` is
required whenever either daily-extreme variable is requested.

Supported variables and final display units are:

| Variable | Meaning | Output unit |
| --- | --- | --- |
| `2t` | Instantaneous 2 m temperature | degrees Celsius |
| `tmax` | Daily maximum 2 m temperature | degrees Celsius |
| `tmin` | Daily minimum 2 m temperature | degrees Celsius |
| `tp` | Total precipitation | millimetres |
| `10u` | 10 m eastward wind | metres/second |
| `10v` | 10 m northward wind | metres/second |

BMA fitting and sampling use the raw model/training units. Conversion to
display units happens only after prediction. The built-in precipitation event
threshold is 1 mm (`0.001` m) and is returned as `probability`.

## Pipeline Result

`run_pipeline` returns a dictionary with four keys:

- `locations`: the input location rows
- `forecast_run`: run ID, creation timestamp, forecast reference time, and
	model version
- `forecasts`: one record per location, variable, and valid time, containing
	`central_value`, `p10`, `p25`, `p50`, `p75`, `p90`, and `probability`
- `map_features`: a GeoJSON `FeatureCollection` for map clients

The pipeline does not write to MySQL, expose an API, or update the frontend.
Those returned structures are intended to be adapted by the surrounding
backend.

### GeoJSON Helpers

`geojson_utils.py` provides the same map representation for both freshly
generated pipeline results and forecast records loaded from the database:

```python
from geojson_utils import build_feature_collection, build_map_feature

feature = build_map_feature(location, forecast_record)
collection = build_feature_collection(locations_by_id, forecast_records)
```

`build_map_feature` returns a GeoJSON `Feature` with a point geometry and
forecast properties. `build_feature_collection` returns a GeoJSON
`FeatureCollection` and skips records whose `location_id` is not present in
`locations_by_id`.

## Direct BMAFuser Usage

For custom integrations, `BMAFuser` can be used without the data-download
pipeline:

```python
from bma_fuser import BMAFuser

fuser = BMAFuser()
fuser.fit_variable("tp", observations, ifs_values, aifs_values)
fuser.save_params("bma_params.json")

fuser.load_params("bma_params.json")
forecast = fuser.predict("tp", ifs_value, aifs_value, event_threshold=0.001)
```

`predict` uses deterministic random sampling when the default random seed is
kept. Change `random_state` when independent sampling streams are required.

## Current Limitations

- `main.py` is only a placeholder greeting and is not the forecast entry
	point.
- `pyproject.toml` does not declare the runtime dependencies listed above.
- AIFS model naming can differ between Herbie releases; adjust
	`MODEL_NAME` in `pull_regrid_aifs.py` if the installed release does not
	recognize `aifs`.
- Forecast retrieval depends on external data availability and may be slow or
	fail for unavailable model runs.
- IMD observations cover an India-only domain. Locations outside that domain
	become missing training values and are excluded from fitting.

## Recommended Run Order

1. Prepare and verify `locations.csv`.
2. Train parameters with `TrainBMA.py` and inspect `bma_params.json`.
3. Ensure every requested forecast variable has fitted parameters.
4. Call `run_pipeline` and persist or serve its returned records in the
	 database/API layer.
