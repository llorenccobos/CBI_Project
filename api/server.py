"""FastAPI application: REST/GeoJSON API + static dashboard."""
from __future__ import annotations

import logging
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import services as S
from config.settings import STATIC_DIR, settings
from models.predict import ModelNotTrained

log = logging.getLogger(__name__)

app = FastAPI(title="BCN Flood & Transport Monitor API", version="0.1.0",
              description="Decision-support prototype. Sensor data are SYNTHETIC.")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except ModelNotTrained as exc:
        raise HTTPException(503, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"not found: {exc}") from exc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def api_meta():
    return S.meta()


@app.get("/api/status")
def api_status():
    return S.status()


@app.get("/api/layers/{name}")
def api_layer(name: str):
    return JSONResponse(_guard(S.layer, name), headers={"Cache-Control": "max-age=3600"})


@app.get("/api/forecast")
def api_forecast(issue: str | None = Query(None, description="ISO time for replay, empty = live")):
    return _guard(S.forecast, issue)


@app.get("/api/sensor/{sensor_id}")
def api_sensor(sensor_id: str, issue: str | None = None):
    return _guard(S.sensor_detail, sensor_id, issue)


@app.get("/api/weather")
def api_weather(issue: str | None = None):
    return _guard(S.weather_layer, issue)


@app.get("/api/events")
def api_events():
    return _guard(S.events)


@app.get("/api/model")
def api_model():
    return S.model_report()


@app.post("/api/refresh")
def api_refresh():
    threading.Thread(target=S.run_update, daemon=True).start()
    return {"ok": True, "message": "update started"}


def start_scheduler():
    cfg = settings()["scheduler"]
    if not cfg.get("enabled", True):
        return None
    from apscheduler.schedulers.background import BackgroundScheduler
    sch = BackgroundScheduler(daemon=True)
    sch.add_job(S.run_update, "interval", minutes=cfg["update_minutes"], id="update",
                max_instances=1, coalesce=True)
    sch.start()
    log.info("scheduler started: update every %d min", cfg["update_minutes"])
    return sch
