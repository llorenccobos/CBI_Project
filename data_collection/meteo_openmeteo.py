"""Open-Meteo weather client (no API key required for non-commercial use).

Three products are used:

* Historical Forecast API  -> continuous, high-resolution hourly series built
  from the first hours of successive model runs (AROME 1.3 km / ICON / IFS).
  Used as the "observed" weather for history and for the synthetic sensors.
* Previous Runs API        -> what the models predicted 1, 2 and 3 days
  before each hour. Used to train the ML model with *realistic forecast
  errors* instead of perfect future rainfall (avoids optimistic evaluation).
* Forecast API             -> live forecast for the next 72 h.

All weather is retrieved on a regular grid over Barcelona (config:
weather.grid), so different districts get different meteorological inputs.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config.settings import RAW_DIR, settings
from data_collection.http import SourceUnavailable, get_json

log = logging.getLogger(__name__)
CACHE = RAW_DIR / "openmeteo"
CACHE.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- grid
def grid_cells() -> pd.DataFrame:
    g = settings()["weather"]["grid"]
    lons = np.linspace(g["lon_min"], g["lon_max"], g["n_lon"])
    lats = np.linspace(g["lat_min"], g["lat_max"], g["n_lat"])
    dlon = (lons[1] - lons[0]) if len(lons) > 1 else 0.03
    dlat = (lats[1] - lats[0]) if len(lats) > 1 else 0.03
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            rows.append({
                "cell_id": f"C{i}{j}", "row": i, "col": j,
                "lat": round(float(lat), 4), "lon": round(float(lon), 4),
                "lon_min": lon - dlon / 2, "lon_max": lon + dlon / 2,
                "lat_min": lat - dlat / 2, "lat_max": lat + dlat / 2,
            })
    return pd.DataFrame(rows)


def upstream_cell(cell_id: str, cells: pd.DataFrame) -> str:
    """Barcelona drains from Collserola (NW) to the sea (SE): upstream = one row north, one col west."""
    c = cells.set_index("cell_id").loc[cell_id]
    r = min(int(c["row"]) + 1, int(cells["row"].max()))
    k = max(int(c["col"]) - 1, 0)
    return cells[(cells.row == r) & (cells.col == k)].cell_id.iloc[0]


# --------------------------------------------------------------------------- parsing
def _to_frame(payload, cells: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = [payload]
    frames = []
    for (_, cell), loc in zip(cells.iterrows(), payload):
        h = loc.get("hourly") or {}
        if "time" not in h:
            continue
        df = pd.DataFrame({"ts": pd.to_datetime(h["time"])})
        for col in columns:
            df[col] = pd.to_numeric(pd.Series(h.get(col, [None] * len(df))), errors="coerce")
        df["cell_id"] = cell["cell_id"]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["cell_id", "ts", *columns])
    return pd.concat(frames, ignore_index=True)


def _coords(cells: pd.DataFrame) -> dict:
    return {"latitude": ",".join(map(str, cells.lat)), "longitude": ",".join(map(str, cells.lon))}


def _om() -> dict:
    return settings()["weather"]["open_meteo"]


# --------------------------------------------------------------------------- history
def _chunks(start: date, end: date, days: int = 183):
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def fetch_history(start: date, end: date, use_cache: bool = True) -> pd.DataFrame:
    """Hourly 'observed' weather (historical forecast stitching) for all grid cells."""
    cells = grid_cells()
    variables = settings()["weather"]["hourly_variables"]
    frames, missing = [], []
    for a, b in _chunks(start, end):
        path = CACHE / f"hist_{a}_{b}.json"
        complete = b < (datetime.now(timezone.utc).date() - timedelta(days=2))
        if use_cache and path.exists():
            payload = json.loads(path.read_text())
        else:
            params = {**_coords(cells), "hourly": ",".join(variables), "start_date": str(a),
                      "end_date": str(b), "timezone": "GMT", "models": _om()["model"]}
            try:
                payload = get_json(_om()["historical_forecast_url"], params,
                                   retries=_om()["max_retries"], wait_s=_om()["retry_wait_s"], timeout=300)
            except SourceUnavailable as exc:
                missing.append(f"{a}..{b}: {str(exc)[:120]}")
                continue
            if complete:  # only cache closed periods
                path.write_text(json.dumps(payload))
        frames.append(_to_frame(payload, cells, variables))
        log.info("weather history %s..%s ok", a, b)
    if not frames:
        raise SourceUnavailable("; ".join(missing))
    df = pd.concat(frames, ignore_index=True)
    df["source"] = "open-meteo:historical-forecast"
    df.attrs["missing_chunks"] = missing
    return df


def fetch_previous_runs(start: date, end: date, use_cache: bool = True) -> pd.DataFrame:
    """Precipitation forecast issued 1, 2 and 3 days before each hour."""
    cells = grid_cells()
    cols = ["precipitation_previous_day1", "precipitation_previous_day2", "precipitation_previous_day3"]
    frames, missing = [], []
    for a, b in _chunks(start, end):
        path = CACHE / f"prev_{a}_{b}.json"
        complete = b < (datetime.now(timezone.utc).date() - timedelta(days=2))
        if use_cache and path.exists():
            payload = json.loads(path.read_text())
        else:
            params = {**_coords(cells), "hourly": ",".join(cols), "start_date": str(a),
                      "end_date": str(b), "timezone": "GMT"}
            try:
                payload = get_json(_om()["previous_runs_url"], params,
                                   retries=_om()["max_retries"], wait_s=_om()["retry_wait_s"], timeout=300)
            except SourceUnavailable as exc:
                missing.append(f"{a}..{b}: {str(exc)[:120]}")
                continue
            if complete:
                path.write_text(json.dumps(payload))
        frames.append(_to_frame(payload, cells, cols))
        log.info("previous runs %s..%s ok", a, b)
    if not frames:
        raise SourceUnavailable("; ".join(missing))
    df = pd.concat(frames, ignore_index=True)
    df.attrs["missing_chunks"] = missing
    return df.rename(columns={cols[0]: "precip_d1", cols[1]: "precip_d2", cols[2]: "precip_d3"})


# --------------------------------------------------------------------------- live forecast
def fetch_forecast() -> tuple[pd.DataFrame, str]:
    """Live forecast (past 2 days + next N days). Returns (frame, source_label).

    Falls back to the Historical Forecast API (which also serves the most
    recent run) if the main forecast endpoint is rate limited.
    """
    cells = grid_cells()
    variables = settings()["weather"]["hourly_variables"]
    w = settings()["weather"]
    params = {**_coords(cells), "hourly": ",".join(variables), "past_days": 2,
              "forecast_days": w["forecast_days"], "timezone": "GMT", "models": _om()["model"]}
    try:
        payload = get_json(_om()["forecast_url"], params, retries=2, wait_s=5)
        label = "open-meteo:forecast"
    except SourceUnavailable as exc:
        log.warning("Forecast API unavailable (%s); trying historical-forecast endpoint", exc)
        today = datetime.now(timezone.utc).date()
        params2 = {**_coords(cells), "hourly": ",".join(variables), "timezone": "GMT",
                   "start_date": str(today - timedelta(days=2)),
                   "end_date": str(today + timedelta(days=w["forecast_days"] - 1)), "models": _om()["model"]}
        payload = get_json(_om()["historical_forecast_url"], params2, retries=2, wait_s=5)
        label = "open-meteo:historical-forecast(latest run)"
    df = _to_frame(payload, cells, variables)
    return df, label


def fetch_elevations(lons: list[float], lats: list[float]) -> list[float | None]:
    try:
        d = get_json(_om()["elevation_url"],
                     {"latitude": ",".join(f"{x:.5f}" for x in lats),
                      "longitude": ",".join(f"{x:.5f}" for x in lons)}, retries=2, wait_s=5)
        return [float(v) for v in d["elevation"]]
    except (SourceUnavailable, KeyError):
        return [None] * len(lons)
