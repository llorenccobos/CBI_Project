"""Central configuration loader.

All modules import configuration from here so that paths, YAML settings,
thresholds and secrets are resolved in a single place.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GEO_DIR = DATA_DIR / "geo"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
REPORTS_DIR = ROOT / "reports"
STATIC_DIR = ROOT / "dashboard" / "static"

for _d in (DATA_DIR, RAW_DIR, GEO_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def settings() -> dict:
    return _load_yaml("settings.yaml")


@lru_cache(maxsize=1)
def locations() -> dict:
    return _load_yaml("locations.yaml")


def thresholds_config() -> dict:
    # Not cached: thresholds can be edited while the app is running.
    return _load_yaml("thresholds.yaml")


LEVEL_ORDER = ["elevated", "dangerous", "critical"]
RISK_CLASSES = ["normal", "elevated", "dangerous", "critical"]


def thresholds_for(sensor_id: str, infra_type: str) -> dict:
    """Return {'elevated': x, 'dangerous': y, 'critical': z} in cm for a sensor."""
    cfg = thresholds_config()
    base = dict(cfg.get("defaults", {}).get(infra_type, {"elevated": 10, "dangerous": 20, "critical": 30}))
    base.update((cfg.get("overrides") or {}).get(sensor_id, {}))
    return {k: float(base[k]) for k in LEVEL_ORDER}


def alert_level() -> str:
    return thresholds_config().get("alert_level", "critical")


def risk_class(level_cm: float, thr: dict) -> str:
    if level_cm is None:
        return "unknown"
    if level_cm >= thr["critical"]:
        return "critical"
    if level_cm >= thr["dangerous"]:
        return "dangerous"
    if level_cm >= thr["elevated"]:
        return "elevated"
    return "normal"


def secret(name: str) -> str | None:
    """Read an API credential from the environment / .env file."""
    value = os.getenv(name, "").strip()
    return value or None


def db_url() -> str:
    url = settings()["storage"]["url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        # make relative SQLite paths relative to the project root
        url = "sqlite:///" + str(ROOT / url[len("sqlite:///"):])
    return url
