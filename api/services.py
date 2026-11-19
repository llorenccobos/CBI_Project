"""Application services used by the HTTP API (kept separate from routing)."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from config.settings import GEO_DIR, REPORTS_DIR, alert_level, settings, thresholds_for
from data_processing.features import LMAX, FeatureStore
from data_storage import db
from data_storage.repository import (last_observation_time, latest_forecast_run, load_observations,
                                     load_sensors, load_weather)
from models import predict as P

log = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()
LAYERS = {"metro_lines", "metro_stations", "bus_routes", "bus_stops", "segments", "weather_grid", "vehicles"}


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _cached(key, fn, maxsize: int = 24):
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    val = fn()
    with _lock:
        _cache[key] = val
        while len(_cache) > maxsize:
            _cache.popitem(last=False)
    return val


def _norm_issue(issue: str | None):
    if not issue or issue == "live":
        return None
    return pd.Timestamp(issue).floor("h")


# --------------------------------------------------------------------------- meta / status
def meta() -> dict:
    cfg = settings()
    sensors = load_sensors()
    thr = {r.sensor_id: thresholds_for(r.sensor_id, r.infra_type) for r in sensors.itertuples()}
    return {
        "project": cfg["project"], "branding": cfg["branding"], "area": cfg["area"],
        "horizons": cfg["forecast"]["horizons_h"], "operational_max_h": cfg["forecast"]["operational_max_h"],
        "alert_level": alert_level(), "thresholds": thr,
        "risk_colors": {"normal": "#2BA84A", "elevated": "#F2C200", "dangerous": "#F07D00", "critical": "#D7191C",
                        "unknown": "#9AA0A6"},
        "vehicles": cfg["vehicles"],
    }


def status() -> dict:
    st = db.get_status()
    last_obs = last_observation_time()
    issued, _ = latest_forecast_run()
    model_v = P.latest_version()
    model_meta = {}
    if model_v:
        try:
            b = P.get_bundle()
            model_meta = {"version": model_v, "splits": b.meta.get("splits"), "data": b.meta.get("data"),
                          "created": b.meta.get("created")}
        except Exception as exc:  # noqa: BLE001
            model_meta = {"version": model_v, "error": str(exc)}
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return {
        "server_time_utc": now.isoformat(),
        "sensor_data": {"last_observation": last_obs.isoformat() if last_obs is not None else None,
                        "age_hours": round((now - last_obs).total_seconds() / 3600, 1) if last_obs is not None else None,
                        "synthetic": True},
        "weather_forecast_run": issued.isoformat() if issued is not None else None,
        "sources": json.loads(st.to_json(orient="records", date_format="iso")) if not st.empty else [],
        "model": model_meta,
    }


def layer(name: str):
    if name not in LAYERS:
        raise KeyError(name)
    ext = "json" if name == "vehicles" else "geojson"
    return json.loads((GEO_DIR / f"{name}.{ext}").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- forecasts
def forecast(issue: str | None) -> dict:
    t = _norm_issue(issue)
    return _cached(("fc", str(t)), lambda: P.forecast(t))


def events(limit: int = 30) -> list[dict]:
    """Past episodes where at least one sensor exceeded its alert threshold (for replay)."""
    def build():
        sensors = load_sensors()
        obs = load_observations()
        w = obs.pivot_table(index="ts", columns="sensor_id", values="level_cm")
        alert = alert_level()
        thr = pd.Series({r.sensor_id: thresholds_for(r.sensor_id, r.infra_type)[alert] for r in sensors.itertuples()})
        exceed = w.ge(thr[w.columns], axis=1)
        n_exc = exceed.sum(axis=1)
        hits = n_exc[n_exc > 0]
        out, cur = [], None
        for ts, n in hits.items():
            if cur is None or ts - cur["end"] > pd.Timedelta(hours=12):
                cur = {"start": ts, "end": ts, "n_sensors": int(n), "peak": float(w.loc[ts].max()),
                       "sensors": set(exceed.columns[exceed.loc[ts]])}
                out.append(cur)
            else:
                cur["end"] = ts
                cur["n_sensors"] = max(cur["n_sensors"], int(n))
                cur["peak"] = max(cur["peak"], float(w.loc[ts].max()))
                cur["sensors"] |= set(exceed.columns[exceed.loc[ts]])
        splits = {}
        try:
            splits = P.get_bundle().meta.get("splits", {})
        except Exception:  # noqa: BLE001
            pass
        cal_end = pd.Timestamp(splits["cal_end"]) if splits.get("cal_end") else None
        tr_end = pd.Timestamp(splits["train_end"]) if splits.get("train_end") else None
        res = []
        for e in out:
            period = "unknown"
            if cal_end is not None:
                period = ("test (out-of-sample)" if e["start"] > cal_end else
                          "training (in-sample)" if e["start"] <= tr_end else "validation/calibration")
            res.append({"start": e["start"].isoformat(), "end": e["end"].isoformat(),
                        "issue_suggested": (e["start"] - pd.Timedelta(hours=3)).isoformat(),
                        "n_sensors": len(e["sensors"]), "peak_cm": round(e["peak"], 1), "period": period})
        res.sort(key=lambda r: (-r["n_sensors"], r["start"]))
        return sorted(res[:limit], key=lambda r: r["start"], reverse=True)
    return _cached(("events",), build)


def sensor_detail(sensor_id: str, issue: str | None) -> dict:
    fc = forecast(issue)
    t0 = pd.Timestamp(fc["issued_at"])
    hist_h = settings()["forecast"]["history_hours_panel"]
    sensors = load_sensors().set_index("sensor_id")
    if sensor_id not in sensors.index:
        raise KeyError(sensor_id)
    s = sensors.loc[sensor_id]
    obs = load_observations(start=t0 - pd.Timedelta(hours=hist_h), end=t0 + pd.Timedelta(hours=LMAX),
                            sensor_id=sensor_id).sort_values("ts")
    past, after = obs[obs.ts <= t0], obs[obs.ts > t0]
    w = load_weather(start=t0 - pd.Timedelta(hours=hist_h), end=t0 + pd.Timedelta(hours=LMAX))
    wl = w[w.cell_id == s.weather_cell].set_index("ts").sort_index()
    rain_past = wl.loc[:t0, "precipitation"]
    rain_fc = forecast_rain_series(s.weather_cell, t0, fc["mode"])
    entry = next(x for x in fc["sensors"] if x["sensor_id"] == sensor_id)
    segs = db.read_sql("SELECT * FROM segments WHERE segment_id = :sid", sid=s.segment_id)
    seg = segs.iloc[0] if len(segs) else None

    def rec(df, col):
        return [{"ts": r.ts.isoformat(), "v": None if pd.isna(getattr(r, col)) else round(float(getattr(r, col)), 2),
                 "q": getattr(r, "quality", None)} for r in df.itertuples()]

    return {
        "sensor": {"sensor_id": sensor_id, "name": s["name"], "infra_type": s.infra_type,
                   "lon": s.lon, "lat": s.lat, "elevation_m": None if pd.isna(s.elevation_m) else s.elevation_m,
                   "weather_cell": s.weather_cell, "upstream_cell": s.upstream_cell,
                   "unit": s.unit, "is_synthetic": bool(s.is_synthetic), "attrs": s.attrs,
                   "lines": json.loads(seg.lines) if seg is not None else [],
                   "stops": json.loads(seg.stops) if seg is not None else [],
                   "segment_length_m": float(seg.length_m) if seg is not None else None},
        "issued_at": fc["issued_at"], "mode": fc["mode"],
        "history": rec(past, "level_cm"),
        "observed_after": rec(after, "level_cm") if fc["mode"] == "replay" else [],
        "rain_history": [{"ts": t.isoformat(), "v": None if pd.isna(v) else round(float(v), 2)}
                         for t, v in rain_past.items()],
        "rain_forecast": rain_fc,
        "forecast": entry,
        "weather_forecast": fc["weather_forecast"],
    }


def forecast_rain_series(cell: str, t0: pd.Timestamp, mode: str) -> list[dict]:
    """Hourly forecast precipitation for the next 72 h used by the model."""
    if mode == "live":
        issued, run = latest_forecast_run()
        if issued is not None and not run.empty:
            r = run[(run.cell_id == cell) & (run.ts > t0)].sort_values("ts")
            return [{"ts": x.ts.isoformat(), "v": None if pd.isna(x.precipitation) else round(float(x.precipitation), 2)}
                    for x in r.itertuples()][:LMAX]
        return []
    store = _replay_store(t0)
    F = store.proxy_forecast_matrix(cell)[-1]
    return [{"ts": (t0 + pd.Timedelta(hours=L + 1)).isoformat(), "v": None if np.isnan(v) else round(float(v), 2)}
            for L, v in enumerate(F)]


def _replay_store(t0):
    return _cached(("store", str(t0)), lambda: FeatureStore(start=t0 - pd.Timedelta(days=2), end=t0), maxsize=24)


def weather_layer(issue: str | None) -> dict:
    """Per-cell weather for 'now' and each horizon: hourly rain at valid time, accumulated rain, T, wind, RH."""
    fc = forecast(issue)
    t0 = pd.Timestamp(fc["issued_at"])
    horizons = [0] + fc["horizons"]

    def build():
        cells = layer("weather_grid")["features"]
        ids = [c["properties"]["cell_id"] for c in cells]
        w = load_weather(start=t0 - pd.Timedelta(hours=24), end=t0 + pd.Timedelta(hours=LMAX))
        obs = {v: w.pivot_table(index="ts", columns="cell_id", values=v) for v in
               ("precipitation", "temperature_2m", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m")}
        src = "observed (analysis)"
        run = None
        if fc["mode"] == "live":
            issued, run = latest_forecast_run()
            src = "live forecast" if run is not None and not run.empty else "no live forecast available"
        else:
            src = "reconstructed from archived forecasts"
            store = _replay_store(t0)
        out = {"issued_at": t0.isoformat(), "source": src, "horizons": {}}
        for h in horizons:
            vt = t0 + pd.Timedelta(hours=h)
            vals = {}
            for cid in ids:
                rec = {}
                if h == 0:
                    for v, df in obs.items():
                        rec[v] = _get(df, vt, cid)
                    rec["acc"] = round(float(obs["precipitation"].loc[t0 - pd.Timedelta(hours=5):t0, cid].sum()), 2) \
                        if cid in obs["precipitation"].columns else None
                elif fc["mode"] == "replay":
                    F = store.proxy_forecast_matrix(cid)[-1]
                    rec["precipitation"] = None if np.isnan(F[h - 1]) else round(float(F[h - 1]), 2)
                    rec["acc"] = round(float(np.nansum(F[:h])), 2)
                    for v in ("temperature_2m", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m"):
                        rec[v] = _get(obs[v], vt, cid)
                elif run is not None and not run.empty:
                    r = run[run.cell_id == cid].set_index("ts")
                    for v in ("precipitation", "temperature_2m", "wind_speed_10m", "wind_direction_10m",
                              "relative_humidity_2m"):
                        rec[v] = _get_series(r[v], vt) if v in r else None
                    win = r.loc[(r.index > t0) & (r.index <= vt), "precipitation"]
                    rec["acc"] = round(float(win.sum()), 2) if len(win) else None
                vals[cid] = rec
            out["horizons"][str(h)] = vals
        return out
    return _cached(("wx", str(t0), fc["mode"]), build)


def _get(df, ts, col):
    try:
        v = df.at[ts, col]
        return None if pd.isna(v) else round(float(v), 2)
    except KeyError:
        return None


def _get_series(s, ts):
    try:
        v = s.loc[ts]
        return None if pd.isna(v) else round(float(v), 2)
    except KeyError:
        return None


def model_report() -> dict:
    f = REPORTS_DIR / "evaluation.json"
    if not f.exists():
        return {"available": False}
    r = json.loads(f.read_text())
    r["available"] = True
    md = REPORTS_DIR / "model_report.md"
    r["markdown"] = md.read_text(encoding="utf-8") if md.exists() else ""
    return r


# --------------------------------------------------------------------------- periodic update
_update_lock = threading.Lock()


def run_update() -> dict:
    """Weather refresh -> extend synthetic sensors -> invalidate forecast cache."""
    from data_collection import weather_service
    from data_collection.sensors_synthetic import generate
    if not _update_lock.acquire(blocking=False):
        return {"ok": False, "message": "update already running"}
    t = time.time()
    try:
        from data_collection import meteo_stations
        res = {"history": weather_service.update_history(), "forecast": weather_service.update_forecast()}
        meteo_stations.fetch_meteocat()
        meteo_stations.fetch_aemet()
        generate()
        clear_cache()
        res["seconds"] = round(time.time() - t, 1)
        log.info("update finished: %s", res)
        return {"ok": True, **res}
    except Exception as exc:  # noqa: BLE001
        log.exception("update failed")
        return {"ok": False, "message": str(exc)}
    finally:
        _update_lock.release()
