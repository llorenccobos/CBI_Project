"""Database layer (SQLAlchemy Core).

SQLite is used for the prototype. All SQL goes through SQLAlchemy so the
storage can be migrated to PostgreSQL/PostGIS by changing `storage.url` in
config/settings.yaml (geometries are stored as GeoJSON text and can be
converted to PostGIS geometry columns in a migration).
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                        create_engine, event, text)

from config.settings import db_url

metadata = MetaData()

sensors = Table(
    "sensors", metadata,
    Column("sensor_id", String, primary_key=True),
    Column("name", String),
    Column("infra_type", String),          # metro | street
    Column("lon", Float), Column("lat", Float),
    Column("segment_id", String),
    Column("weather_cell", String),
    Column("upstream_cell", String),
    Column("elevation_m", Float),
    Column("vulnerability", String),
    Column("unit", String, default="cm"),
    Column("is_synthetic", Boolean, default=True),
    Column("attrs", Text),                 # JSON: lines, stations, notes ...
)

segments = Table(
    "segments", metadata,
    Column("segment_id", String, primary_key=True),
    Column("infra_type", String),
    Column("name", String),
    Column("geometry", Text),              # GeoJSON LineString (EPSG:4326)
    Column("length_m", Float),
    Column("lines", Text),                 # JSON list of affected lines
    Column("stops", Text),                 # JSON list of connected stations / stops
)

observations = Table(
    "sensor_observations", metadata,
    Column("sensor_id", String, primary_key=True),
    Column("ts", DateTime, primary_key=True),       # UTC, hourly
    Column("level_cm", Float),
    Column("quality", String),                      # good | suspect | missing
    Column("is_synthetic", Boolean),
)

weather_hourly = Table(
    "weather_hourly", metadata,
    Column("cell_id", String, primary_key=True),
    Column("ts", DateTime, primary_key=True),
    Column("precipitation", Float), Column("temperature_2m", Float),
    Column("relative_humidity_2m", Float), Column("surface_pressure", Float),
    Column("wind_speed_10m", Float), Column("wind_direction_10m", Float),
    Column("wind_gusts_10m", Float), Column("cape", Float),
    Column("precip_d1", Float), Column("precip_d2", Float), Column("precip_d3", Float),
    Column("source", String),
)

weather_forecast = Table(
    "weather_forecast", metadata,
    Column("issued_at", DateTime, primary_key=True),
    Column("cell_id", String, primary_key=True),
    Column("ts", DateTime, primary_key=True),
    Column("precipitation", Float), Column("temperature_2m", Float),
    Column("relative_humidity_2m", Float), Column("surface_pressure", Float),
    Column("wind_speed_10m", Float), Column("wind_direction_10m", Float),
    Column("wind_gusts_10m", Float), Column("cape", Float),
    Column("source", String),
)

predictions = Table(
    "predictions", metadata,
    Column("issued_at", DateTime, primary_key=True),
    Column("sensor_id", String, primary_key=True),
    Column("horizon_h", Integer, primary_key=True),
    Column("payload", Text),               # JSON with quantiles, probabilities, confidence ...
    Column("model_version", String),
)

source_status = Table(
    "source_status", metadata,
    Column("source", String, primary_key=True),
    Column("status", String),              # ok | degraded | unavailable | not_configured
    Column("last_success", DateTime),
    Column("last_attempt", DateTime),
    Column("message", Text),
)


@lru_cache(maxsize=1)
def engine():
    eng = create_engine(db_url(), future=True)
    if eng.dialect.name == "sqlite":
        @event.listens_for(eng, "connect")
        def _pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return eng


def init_db() -> None:
    metadata.create_all(engine())


def upsert_df(table: Table, df: pd.DataFrame, chunk: int = 20000) -> int:
    """Insert-or-replace a DataFrame into `table` (columns not in the table are ignored)."""
    if df is None or df.empty:
        return 0
    cols = [c.name for c in table.columns if c.name in df.columns]
    data = df[cols].astype(object).where(pd.notna(df[cols]), None)
    records = data.to_dict("records")
    for rec in records:  # pandas Timestamps -> python datetimes
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                rec[k] = v.to_pydatetime()
    eng = engine()
    with eng.begin() as conn:
        for i in range(0, len(records), chunk):
            batch = records[i:i + chunk]
            if eng.dialect.name == "sqlite":
                stmt = table.insert().prefix_with("OR REPLACE")
            else:  # PostgreSQL
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                ins = pg_insert(table)
                pk = [c.name for c in table.primary_key.columns]
                stmt = ins.on_conflict_do_update(index_elements=pk,
                                                 set_={c: ins.excluded[c] for c in cols if c not in pk})
            conn.execute(stmt, batch)
    return len(records)


def read_sql(query: str, **params) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def set_status(source: str, status: str, message: str = "", success: bool | None = None) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    prev = read_sql("SELECT last_success FROM source_status WHERE source=:s", s=source)
    last_success = prev.last_success.iloc[0] if len(prev) else None
    if success or (success is None and status == "ok"):
        last_success = now
    upsert_df(source_status, pd.DataFrame([{
        "source": source, "status": status, "message": message[:500],
        "last_attempt": now, "last_success": pd.to_datetime(last_success) if last_success is not None else None,
    }]))


def get_status() -> pd.DataFrame:
    return read_sql("SELECT * FROM source_status")
