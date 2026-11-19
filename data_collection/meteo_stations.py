"""Optional station observations: Meteocat XEMA and AEMET OpenData.

Both providers are free but require an API key:
  * METEOCAT_API_KEY  -> request at https://apidocs.meteo.cat (header X-Api-Key)
  * AEMET_API_KEY     -> request at https://opendata.aemet.es/centrodedescargas/altaUsuario

They complement the Open-Meteo grid with *measured* rainfall at a few points
in Barcelona (useful for bias checks and, once enough history is stored, as
additional model inputs). If no key is configured the module does nothing and
the data-source status shows "not_configured".

NOTE: these clients follow the providers' public documentation but could not
be tested without keys during development; check the status panel after
adding your keys.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import Column, DateTime, Float, String, Table

from config.settings import secret, settings
from data_collection.http import SourceUnavailable, get_json
from data_storage import db

log = logging.getLogger(__name__)

station_obs = Table(
    "station_observations", db.metadata,
    Column("source", String, primary_key=True),
    Column("station_id", String, primary_key=True),
    Column("variable", String, primary_key=True),
    Column("ts", DateTime, primary_key=True),
    Column("value", Float),
)

# Meteocat XEMA variable codes
METEOCAT_VARS = {"precipitation": 35, "temperature_2m": 32, "relative_humidity_2m": 33,
                 "surface_pressure": 34, "wind_speed_10m": 30, "wind_direction_10m": 31}


def fetch_meteocat(day: date | None = None) -> pd.DataFrame:
    key = secret("METEOCAT_API_KEY")
    if not key:
        db.set_status("meteocat:xema", "not_configured", "set METEOCAT_API_KEY in .env", success=False)
        return pd.DataFrame()
    cfg = settings()["weather"]["meteocat"]
    day = day or datetime.now(timezone.utc).date()
    rows = []
    try:
        for var, code in METEOCAT_VARS.items():
            for st in cfg["stations"]:
                url = f"{cfg['base_url']}/variables/mesurades/{code}/{day:%Y/%m/%d}"
                data = get_json(url, {"codiEstacio": st}, headers={"X-Api-Key": key}, retries=2, wait_s=3)
                for est in data if isinstance(data, list) else [data]:
                    for v in est.get("variables", []):
                        for lec in v.get("lectures", []):
                            rows.append({"source": "meteocat", "station_id": est.get("codi", st), "variable": var,
                                         "ts": pd.Timestamp(lec["data"]).tz_convert(None) if "Z" in lec["data"]
                                         else pd.Timestamp(lec["data"]), "value": lec.get("valor")})
        df = pd.DataFrame(rows)
        db.init_db()
        db.upsert_df(station_obs, df)
        db.set_status("meteocat:xema", "ok", f"{len(df)} readings for {day}")
        return df
    except SourceUnavailable as exc:
        db.set_status("meteocat:xema", "unavailable", str(exc), success=False)
        return pd.DataFrame()


def fetch_aemet() -> pd.DataFrame:
    """Last 24 h of hourly observations for the configured AEMET stations."""
    key = secret("AEMET_API_KEY")
    if not key:
        db.set_status("aemet:opendata", "not_configured", "set AEMET_API_KEY in .env", success=False)
        return pd.DataFrame()
    cfg = settings()["weather"]["aemet"]
    rows = []
    try:
        for st in cfg["stations"]:
            meta = get_json(f"{cfg['base_url']}/observacion/convencional/datos/estacion/{st}",
                            {"api_key": key}, retries=2, wait_s=3)
            for r in get_json(meta["datos"], retries=2, wait_s=3):
                ts = pd.Timestamp(r["fint"])
                for var, col in (("precipitation", "prec"), ("temperature_2m", "ta"),
                                 ("relative_humidity_2m", "hr"), ("surface_pressure", "pres"),
                                 ("wind_speed_10m", "vv")):
                    if col in r and r[col] is not None:
                        val = float(r[col]) * (3.6 if col == "vv" else 1.0)  # m/s -> km/h
                        rows.append({"source": "aemet", "station_id": st, "variable": var, "ts": ts, "value": val})
        df = pd.DataFrame(rows)
        db.init_db()
        db.upsert_df(station_obs, df)
        db.set_status("aemet:opendata", "ok", f"{len(df)} readings")
        return df
    except (SourceUnavailable, KeyError) as exc:
        db.set_status("aemet:opendata", "unavailable", str(exc), success=False)
        return pd.DataFrame()
