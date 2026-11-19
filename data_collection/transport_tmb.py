"""Optional: official TMB GTFS (static timetables, stops and shapes).

Requires free credentials from https://developer.tmb.cat (create an app to
get TMB_APP_ID and TMB_APP_KEY, then put them in .env).

The prototype builds the network from OpenStreetMap (real geometry, no key
needed). When TMB credentials are configured, `download_gtfs()` stores the
official GTFS under data/raw/tmb/ and `gtfs_summary()` exposes routes, stops
and shapes, which can be used to (a) validate OSM geometry and (b) give the
simulated vehicles real timetable headways.

TMB does not publish public real-time vehicle positions (iBus/iMetro provide
arrival predictions only), so vehicles on the map remain SIMULATED.
"""
from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

from config.settings import RAW_DIR, secret, settings
from data_storage import db

log = logging.getLogger(__name__)
TMB_DIR = RAW_DIR / "tmb"


def credentials() -> tuple[str, str] | None:
    app_id, app_key = secret("TMB_APP_ID"), secret("TMB_APP_KEY")
    return (app_id, app_key) if app_id and app_key else None


def download_gtfs() -> bool:
    cred = credentials()
    if not cred:
        db.set_status("tmb:gtfs", "not_configured", "set TMB_APP_ID and TMB_APP_KEY in .env", success=False)
        return False
    url = settings()["transport"]["tmb_gtfs_url"]
    try:
        r = requests.get(url, params={"app_id": cred[0], "app_key": cred[1]}, timeout=120)
        r.raise_for_status()
        TMB_DIR.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(TMB_DIR)
        db.set_status("tmb:gtfs", "ok", f"GTFS downloaded ({len(r.content) // 1024} kB)")
        return True
    except (requests.RequestException, zipfile.BadZipFile) as exc:
        db.set_status("tmb:gtfs", "unavailable", str(exc), success=False)
        return False


def gtfs_summary() -> dict:
    if not (TMB_DIR / "routes.txt").exists():
        return {}
    routes = pd.read_csv(TMB_DIR / "routes.txt", dtype=str)
    stops = pd.read_csv(TMB_DIR / "stops.txt", dtype=str)
    shapes = pd.read_csv(TMB_DIR / "shapes.txt") if (TMB_DIR / "shapes.txt").exists() else pd.DataFrame()
    return {"routes": len(routes), "stops": len(stops), "shapes": shapes["shape_id"].nunique() if len(shapes) else 0}
