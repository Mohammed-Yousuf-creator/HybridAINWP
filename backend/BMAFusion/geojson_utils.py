"""
geojson_utils.py

Tiny shared helper so a forecast record -> map feature turns into the
exact same GeoJSON shape whether it comes fresh out of run_pipeline()
or is read back out of the `forecasts`/`locations` tables later by a
FastAPI endpoint. Keeping this in one place means the frontend's map
code never has to care which path the data took.
"""

from __future__ import annotations


def build_map_feature(location: dict, forecast_record: dict) -> dict:
    """
    location : {"location_id", "name", "latitude", "longitude", ...}
    forecast_record : one row shaped like the `forecasts` table
                       (must include variable, unit, valid_time,
                       central_value, p10, p25, p50, p75, p90, probability)
    """
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [location["longitude"], location["latitude"]]},
        "properties": {
            "location_id": location["location_id"],
            "name": location.get("name"),
            "variable": forecast_record["variable"],
            "unit": forecast_record["unit"],
            "valid_time": forecast_record["valid_time"],
            "central_value": forecast_record["central_value"],
            "p10": forecast_record["p10"],
            "p25": forecast_record["p25"],
            "p50": forecast_record["p50"],
            "p75": forecast_record["p75"],
            "p90": forecast_record["p90"],
            "probability": forecast_record["probability"],
        },
    }


def build_feature_collection(locations_by_id: dict, forecast_records: list[dict]) -> dict:
    """locations_by_id: {location_id: location_dict}, e.g. built once from a DB query."""
    features = [
        build_map_feature(locations_by_id[r["location_id"]], r)
        for r in forecast_records
        if r["location_id"] in locations_by_id
    ]
    return {"type": "FeatureCollection", "features": features}