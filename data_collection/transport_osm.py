"""Download real transport geometry from OpenStreetMap (Overpass API).

* TMB metro route relations (track geometry + station nodes)
* TMB bus route relations (route geometry + stop/platform nodes)
* Street ways around each monitored street location

Data © OpenStreetMap contributors, ODbL. Raw responses are cached under
data/raw/osm/ so that the network only has to be downloaded once.
"""
from __future__ import annotations

import logging

from config.settings import RAW_DIR, locations, settings
from data_collection.http import SourceUnavailable, cached_json, post_form

log = logging.getLogger(__name__)
OSM_DIR = RAW_DIR / "osm"

AREA = 'area["name"="Barcelona"]["admin_level"="8"]->.a;'


def _overpass(query: str, timeout: float = 300, rounds: int = 4) -> dict:
    """Run an Overpass query, cycling through the configured public endpoints.

    Public Overpass servers are frequently overloaded (HTTP 504/429), so each
    endpoint is tried once per round, for several rounds.
    """
    last = None
    for rnd in range(rounds):
        for endpoint in settings()["transport"]["overpass_endpoints"]:
            try:
                log.info("Overpass query via %s (round %d)", endpoint, rnd + 1)
                return post_form(endpoint, {"data": query}, retries=1, wait_s=3, timeout=timeout)
            except SourceUnavailable as exc:
                last = exc
                log.warning("Overpass endpoint failed: %s", str(exc)[:120])
    raise SourceUnavailable(f"All Overpass endpoints failed: {last}")


def fetch_metro(refresh: bool = False) -> dict:
    q = ("[out:json][timeout:240];" + AREA +
         'rel(area.a)["route"="subway"]["network"="Metro de Barcelona"]->.r;'
         ".r out geom; node(r.r); out;")
    return cached_json(OSM_DIR / "metro.json", lambda: _overpass(q), refresh)


def fetch_bus(refresh: bool = False) -> dict:
    op = settings()["transport"]["bus_operator"]
    q = ("[out:json][timeout:300];" + AREA +
         f'rel(area.a)["route"="bus"]["operator"="{op}"]->.r;'
         ".r out geom; node(r.r); out;")
    return cached_json(OSM_DIR / "bus.json", lambda: _overpass(q), refresh)


def fetch_streets(refresh: bool = False, offline: bool = False) -> dict:
    """Street axes around each monitored street location (one small query per street)."""
    elements, seen = [], set()
    for loc in locations()["street_sections"]:
        lon, lat = loc["point"]
        name = loc["street"].replace('"', '\\"')
        bbox = f"{lat - 0.0032:.5f},{lon - 0.0042:.5f},{lat + 0.0032:.5f},{lon + 0.0042:.5f}"
        q = f'[out:json][timeout:60];way["highway"]["name"="{name}"]({bbox});out geom;'
        path = OSM_DIR / f"street_{loc['id']}.json"
        if offline and not path.exists():
            continue
        try:
            data = cached_json(path,
                               lambda q=q: _overpass(q, timeout=45, rounds=6), refresh)
        except SourceUnavailable as exc:  # the network builder falls back to bus-route geometry
            log.warning("street %s not downloaded: %s", loc["id"], exc)
            continue
        for e in data.get("elements", []):
            if e["id"] not in seen:
                seen.add(e["id"])
                elements.append(e)
    return {"elements": elements}


def fetch_all(refresh: bool = False) -> None:
    fetch_metro(refresh)
    fetch_bus(refresh)
    fetch_streets(refresh)
