"""Read helpers returning tidy pandas objects for the processing / ML layers."""
from __future__ import annotations

import json

import pandas as pd

from data_storage.db import read_sql

WEATHER_VARS = ["precipitation", "temperature_2m", "relative_humidity_2m", "surface_pressure",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cape"]


def load_sensors() -> pd.DataFrame:
    df = read_sql("SELECT * FROM sensors ORDER BY sensor_id")
    df["attrs"] = df["attrs"].apply(lambda s: json.loads(s) if s else {})
    return df


def load_segments() -> pd.DataFrame:
    df = read_sql("SELECT * FROM segments")
    for c in ("geometry", "lines", "stops"):
        df[c] = df[c].apply(lambda s: json.loads(s) if s else None)
    return df


def load_observations(start=None, end=None, sensor_id: str | None = None) -> pd.DataFrame:
    q = "SELECT sensor_id, ts, level_cm, quality, is_synthetic FROM sensor_observations WHERE 1=1"
    p = {}
    if start is not None:
        q += " AND ts >= :start"; p["start"] = pd.Timestamp(start).to_pydatetime()
    if end is not None:
        # exclusive upper bound + 1 s: robust to how the DB driver serialises datetimes
        q += " AND ts < :end"; p["end"] = (pd.Timestamp(end) + pd.Timedelta(seconds=1)).to_pydatetime()
    if sensor_id:
        q += " AND sensor_id = :sid"; p["sid"] = sensor_id
    df = read_sql(q, **p)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load_weather(start=None, end=None) -> pd.DataFrame:
    q = "SELECT * FROM weather_hourly WHERE 1=1"
    p = {}
    if start is not None:
        q += " AND ts >= :start"; p["start"] = pd.Timestamp(start).to_pydatetime()
    if end is not None:
        # exclusive upper bound + 1 s: robust to how the DB driver serialises datetimes
        q += " AND ts < :end"; p["end"] = (pd.Timestamp(end) + pd.Timedelta(seconds=1)).to_pydatetime()
    df = read_sql(q, **p)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def latest_forecast_run() -> tuple[pd.Timestamp | None, pd.DataFrame]:
    last = read_sql("SELECT MAX(issued_at) AS t FROM weather_forecast")
    if last.empty or last.t.iloc[0] is None:
        return None, pd.DataFrame()
    t = pd.Timestamp(last.t.iloc[0])
    # range match: robust to how the DB driver serialises datetimes
    df = read_sql("SELECT * FROM weather_forecast WHERE issued_at >= :a AND issued_at < :b",
                  a=(t - pd.Timedelta(seconds=1)).to_pydatetime(), b=(t + pd.Timedelta(seconds=1)).to_pydatetime())
    df["ts"] = pd.to_datetime(df["ts"])
    return t, df


def last_observation_time() -> pd.Timestamp | None:
    d = read_sql("SELECT MAX(ts) AS t FROM sensor_observations WHERE level_cm IS NOT NULL")
    return pd.Timestamp(d.t.iloc[0]) if len(d) and d.t.iloc[0] is not None else None
