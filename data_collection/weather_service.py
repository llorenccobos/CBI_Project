"""Weather ingestion: history + archived forecasts + live forecast -> database.

Source priority: Open-Meteo (gridded, no key) -> local seed snapshot.
Meteocat / AEMET station observations are optional add-ons (see their modules).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from config.settings import DATA_DIR, settings
from data_collection import meteo_openmeteo as om
from data_collection.http import SourceUnavailable
from data_storage import db

log = logging.getLogger(__name__)
SEED = DATA_DIR / "seed" / "weather_hourly.parquet"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def load_seed() -> int:
    """Load the bundled weather snapshot (lets the app run without internet)."""
    if not SEED.exists():
        return 0
    df = pd.read_parquet(SEED)
    n = db.upsert_df(db.weather_hourly, df)
    log.info("loaded %d weather rows from seed snapshot", n)
    return n


def export_seed() -> None:
    from data_storage.repository import load_weather
    df = load_weather()
    SEED.parent.mkdir(parents=True, exist_ok=True)
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")
    df.to_parquet(SEED, compression="zstd", index=False)


def update_history(start: date | None = None) -> dict:
    """Fetch (cached) history + previous runs from Open-Meteo and upsert them."""
    db.init_db()
    start = start or date.fromisoformat(settings()["weather"]["history_start"])
    end = _utcnow().date()
    result = {}
    hist = None
    try:
        hist = om.fetch_history(start, end)
        miss = hist.attrs.get("missing_chunks", [])
        db.set_status("open-meteo:history", "degraded" if miss else "ok",
                      f"{len(hist)} rows up to {hist.ts.max()}" + (f"; missing: {miss}" if miss else ""),
                      success=True)
        result["history_rows"] = len(hist)
    except SourceUnavailable as exc:
        log.warning("history fetch incomplete: %s", exc)
        db.set_status("open-meteo:history", "degraded", str(exc), success=False)
        if db.read_sql("SELECT COUNT(*) n FROM weather_hourly").n.iloc[0] == 0:
            result["seed_rows"] = load_seed()
    if hist is not None and len(hist):
        try:
            prev = om.fetch_previous_runs(start, end)
            hist = hist.merge(prev, on=["cell_id", "ts"], how="left")
            db.set_status("open-meteo:previous-runs", "ok", f"{prev.precip_d1.notna().sum()} rows")
        except SourceUnavailable as exc:
            db.set_status("open-meteo:previous-runs", "degraded", str(exc), success=False)
        db.upsert_df(db.weather_hourly, hist)
    return result


def update_forecast() -> dict:
    """Fetch the live forecast and store it as a forecast run issued now (floored to the hour)."""
    db.init_db()
    try:
        df, label = om.fetch_forecast()
    except SourceUnavailable as exc:
        db.set_status("open-meteo:forecast", "unavailable", str(exc), success=False)
        log.warning("live forecast unavailable: %s", exc)
        return {"ok": False, "error": str(exc)}
    now = pd.Timestamp(_utcnow()).floor("h")
    df = df.dropna(subset=["precipitation"])
    past = df[df.ts <= now].copy()
    past["source"] = label
    # Recent hours from the forecast endpoint complete the observed history
    # (only where the history endpoint has no value yet).
    existing = db.read_sql("SELECT cell_id, ts FROM weather_hourly WHERE ts >= :t AND precipitation IS NOT NULL",
                           t=(now - pd.Timedelta(days=3)).to_pydatetime())
    existing["ts"] = pd.to_datetime(existing["ts"])
    past = past.merge(existing.assign(_e=1), on=["cell_id", "ts"], how="left")
    past = past[past._e.isna()].drop(columns="_e")
    db.upsert_df(db.weather_hourly, past)
    fut = df[df.ts > now].copy()
    fut["issued_at"] = now
    fut["source"] = label
    db.upsert_df(db.weather_forecast, fut)
    status = "ok" if label == "open-meteo:forecast" else "degraded"
    db.set_status("open-meteo:forecast", status, f"{label}: run {now}, {fut.ts.nunique()} h ahead", success=True)
    return {"ok": True, "issued_at": str(now), "hours": int(fut.ts.nunique()), "source": label}
