"""Build the processed transport network and the monitored infrastructure segments.

Inputs : raw OSM responses (data/raw/osm/*.json), config/locations.yaml
Outputs: data/geo/*.geojson (map layers), data/geo/vehicles.json (paths for
         the simulated vehicles), and the `sensors` / `segments` DB tables.

Geometric operations are done in ETRS89 / UTM 31N (EPSG:25831, metres) and
results are stored in WGS84 (EPSG:4326).
"""
from __future__ import annotations

import json
import logging
import unicodedata

import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, mapping
from shapely.ops import linemerge, substring, transform

from config.settings import GEO_DIR, locations, settings
from data_collection import transport_osm
from data_collection.meteo_openmeteo import fetch_elevations, grid_cells, upstream_cell
from data_storage import db

log = logging.getLogger(__name__)

_TO_M = Transformer.from_crs(4326, 25831, always_xy=True).transform
_TO_LL = Transformer.from_crs(25831, 4326, always_xy=True).transform

REF_ALIASES = {"L10 Nord": "L10N", "L10 Sud": "L10S", "L9 Nord": "L9N", "L9 Sud": "L9S"}
BUS_COLOURS = {"H": "#0066B3", "V": "#00A651", "D": "#7B2C83", "X": "#E2001A", "N": "#1D1D1B"}


# --------------------------------------------------------------------------- helpers
def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("’", "'").split())


def to_m(geom):
    return transform(_TO_M, geom)


def to_ll(geom):
    return transform(_TO_LL, geom)


def stitch(ways: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Concatenate ordered relation ways into a single coordinate sequence."""
    coords: list = []
    for w in ways:
        if len(w) < 2:
            continue
        if not coords:
            coords = list(w)
            continue
        d = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2  # noqa: E731
        s, e = coords[0], coords[-1]
        best = min((d(e, w[0]), 0), (d(e, w[-1]), 1), (d(s, w[-1]), 2), (d(s, w[0]), 3))[1]
        if best == 0:
            coords += w[1:]
        elif best == 1:
            coords += w[::-1][1:]
        elif best == 2:
            coords = w[:-1] + coords
        else:
            coords = w[::-1][:-1] + coords
    return coords


def _way_coords(member) -> list[tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in member.get("geometry", []) if p]


def _round_geom(geojson: dict, nd: int = 5) -> dict:
    def r(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    return {"type": geojson["type"], "coordinates": r(geojson["coordinates"])}


def _feature(geom, props: dict) -> dict:
    return {"type": "Feature", "geometry": _round_geom(mapping(geom)), "properties": props}


def _write(name: str, features: list) -> None:
    (GEO_DIR / name).write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                           ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- metro
def build_metro():
    raw = transport_osm.fetch_metro()
    tmb = set(settings()["transport"]["tmb_metro_lines"])
    tol = settings()["transport"]["simplify_tolerance_m"]
    nodes = {e["id"]: e for e in raw["elements"] if e["type"] == "node"}
    lines, stations = {}, {}
    for rel in raw["elements"]:
        if rel["type"] != "relation":
            continue
        ref = REF_ALIASES.get(rel["tags"].get("ref"), rel["tags"].get("ref"))
        if ref not in tmb or ref in lines:  # keep one direction per line
            continue
        ways = [_way_coords(m) for m in rel["members"] if m["type"] == "way" and m["role"] in ("", "route")]
        coords = stitch(ways)
        if len(coords) < 2:
            continue
        geom_m = LineString([_TO_M(*c) for c in coords])
        seq = []
        for m in rel["members"]:
            if m["type"] == "node" and m["role"].startswith("stop"):
                n = nodes.get(m["ref"])
                if not n or "name" not in n.get("tags", {}):
                    continue
                name = n["tags"]["name"]
                if seq and norm(seq[-1][0]) == norm(name):
                    continue
                seq.append((name, n["lon"], n["lat"]))
                st = stations.setdefault(norm(name), {"name": name, "lines": [], "pts": []})
                if ref not in st["lines"]:
                    st["lines"].append(ref)
                st["pts"].append((n["lon"], n["lat"]))
        lines[ref] = {"ref": ref, "name": rel["tags"].get("name", ref),
                      "colour": rel["tags"].get("colour", "#888888"),
                      "geom_m": geom_m, "geom_m_simpl": geom_m.simplify(tol), "stations": seq}
    log.info("metro: %d lines, %d stations", len(lines), len(stations))
    return lines, stations


# --------------------------------------------------------------------------- bus
def build_bus():
    raw = transport_osm.fetch_bus()
    tol = settings()["transport"]["simplify_tolerance_m"]
    nodes = {e["id"]: e for e in raw["elements"] if e["type"] == "node"}
    routes, stops = {}, {}
    for rel in raw["elements"]:
        if rel["type"] != "relation":
            continue
        ref = rel["tags"].get("ref")
        if not ref:
            continue
        ways = [_way_coords(m) for m in rel["members"] if m["type"] == "way" and m["role"] in ("", "route")]
        coords = stitch(ways)
        if len(coords) < 2:
            continue
        geom_m = LineString([_TO_M(*c) for c in coords]).simplify(tol)
        for m in rel["members"]:
            if m["type"] == "node" and (m["role"].startswith("platform") or m["role"].startswith("stop")):
                n = nodes.get(m["ref"])
                if not n:
                    continue
                st = stops.setdefault(n["id"], {"id": n["id"], "name": n.get("tags", {}).get("name", "Bus stop"),
                                                "lon": n["lon"], "lat": n["lat"], "lines": set()})
                st["lines"].add(ref)
        colour = rel["tags"].get("colour") or BUS_COLOURS.get(ref[0], "#E2001A")
        # keep the longest direction for display; both are similar
        if ref not in routes or geom_m.length > routes[ref]["geom_m"].length:
            routes[ref] = {"ref": ref, "name": rel["tags"].get("name", ref), "colour": colour, "geom_m": geom_m}
    log.info("bus: %d routes, %d stops", len(routes), len(stops))
    return routes, stops


# --------------------------------------------------------------------------- segments
def _metro_section(loc, lines):
    line = lines[loc["line"]]
    by_name = {norm(n): (lon, lat) for n, lon, lat in line["stations"]}
    a, b = by_name.get(norm(loc["from"])), by_name.get(norm(loc["to"]))
    if a is None or b is None:
        raise ValueError(f"{loc['id']}: stations {loc['from']} / {loc['to']} not found on {loc['line']}")
    g = line["geom_m"]
    da, dbb = g.project(Point(_TO_M(*a))), g.project(Point(_TO_M(*b)))
    seg = substring(g, min(da, dbb), max(da, dbb))
    if seg.length < 50:  # fall back to a straight connection if projection failed
        seg = LineString([_TO_M(*a), _TO_M(*b)])
    return seg, [loc["from"], loc["to"]]


def _street_section(loc, street_ways, routes):
    half = settings()["transport"]["street_segment_half_length_m"]
    p = Point(_TO_M(*loc["point"]))
    cands = [LineString([_TO_M(*c) for c in w["coords"]]) for w in street_ways
             if norm(w["name"]) == norm(loc["street"]) and len(w["coords"]) > 1]
    source = "osm-street"
    merged = None
    if cands:
        m = linemerge(MultiLineString(cands)) if len(cands) > 1 else cands[0]
        parts = list(m.geoms) if hasattr(m, "geoms") else [m]
        merged = min(parts, key=lambda g: g.distance(p))
        if merged.distance(p) > 250:
            merged = None
    if merged is None:  # fallback: nearest bus route geometry (buses run along these streets)
        source = "bus-route-geometry"
        merged = min((r["geom_m"] for r in routes.values()), key=lambda g: g.distance(p))
    d = merged.project(p)
    seg = substring(merged, max(0, d - half), min(merged.length, d + half))
    return seg, source


def build_segments(lines, stations, routes, stops, offline_streets: bool = False):
    tcfg = settings()["transport"]
    raw_streets = transport_osm.fetch_streets(offline=offline_streets)
    street_ways = [{"name": e.get("tags", {}).get("name", ""), "coords": _way_coords(e)}
                   for e in raw_streets.get("elements", []) if e["type"] == "way"]
    cells = grid_cells()
    seg_rows, sensor_rows = [], []
    loc_cfg = locations()

    def nearest_cell(lon, lat):
        d = (cells.lon - lon) ** 2 + ((cells.lat - lat) * 1.33) ** 2
        return cells.loc[d.idxmin(), "cell_id"]

    stop_pts = [(s, Point(_TO_M(s["lon"], s["lat"]))) for s in stops.values()]

    for loc in loc_cfg.get("metro_sections", []):
        seg_m, ends = _metro_section(loc, lines)
        affected = [loc["line"]]
        for ref, ln in lines.items():  # shared track sections (several lines on the same segment)
            if ref != loc["line"] and seg_m.intersection(ln["geom_m"].buffer(15)).length > 0.5 * seg_m.length:
                affected.append(ref)
        conn = [{"name": stations[norm(n)]["name"], "kind": "metro_station",
                 "lines": stations[norm(n)]["lines"]} for n in ends]
        seg_rows.append(("metro", loc, seg_m, affected, conn,
                         f"{loc['line']} {ends[0]} – {ends[1]}", "osm-metro-track"))

    for loc in loc_cfg.get("street_sections", []):
        seg_m, src = _street_section(loc, street_ways, routes)
        buf = seg_m.buffer(tcfg["bus_buffer_m"])
        affected = sorted([ref for ref, r in routes.items()
                           if r["geom_m"].intersection(buf).length >= tcfg["bus_overlap_min"] * seg_m.length],
                          key=lambda x: (len(x), x))
        near = [s for s, pt in stop_pts if pt.distance(seg_m) < 60 and s["lines"] & set(affected)]
        conn = [{"name": s["name"], "kind": "bus_stop", "lines": sorted(s["lines"] & set(affected))}
                for s in sorted(near, key=lambda s: s["name"])][:8]
        seg_rows.append(("street", loc, seg_m, affected, conn, loc["street"], src))

    mids = [to_ll(seg_m.interpolate(0.5, normalized=True)) for _, _, seg_m, *_ in seg_rows]
    elev = fetch_elevations([p.x for p in mids], [p.y for p in mids])

    seg_features = []
    for (kind, loc, seg_m, affected, conn, name, src), mid, z in zip(seg_rows, mids, elev):
        seg_ll = to_ll(seg_m)
        sid = loc["id"]
        cell = nearest_cell(mid.x, mid.y)
        props = {"segment_id": f"SEG-{sid}", "sensor_id": sid, "infra_type": kind, "name": name,
                 "lines": affected, "stops": conn, "length_m": round(seg_m.length, 1),
                 "geometry_source": src, "note": loc.get("note", "")}
        seg_features.append(_feature(seg_ll, props))
        db.upsert_df(db.segments, pd.DataFrame([{
            "segment_id": props["segment_id"], "infra_type": kind, "name": name,
            "geometry": json.dumps(_round_geom(mapping(seg_ll))), "length_m": props["length_m"],
            "lines": json.dumps(affected), "stops": json.dumps(conn, ensure_ascii=False)}]))
        db.upsert_df(db.sensors, pd.DataFrame([{
            "sensor_id": sid, "name": name, "infra_type": kind, "lon": round(mid.x, 6), "lat": round(mid.y, 6),
            "segment_id": props["segment_id"], "weather_cell": cell, "upstream_cell": upstream_cell(cell, cells),
            "elevation_m": z, "vulnerability": loc.get("vulnerability", "medium"), "unit": "cm",
            "is_synthetic": True,
            "attrs": json.dumps({"note": loc.get("note", ""), "geometry_source": src}, ensure_ascii=False)}]))
        sensor_rows.append(sid)
    _write("segments.geojson", seg_features)
    return seg_features


# --------------------------------------------------------------------------- vehicles
def build_vehicle_paths(lines, routes, seg_features):
    vcfg = settings()["vehicles"]
    out = []
    for ref, ln in lines.items():
        g = to_ll(ln["geom_m_simpl"])
        out.append({"id": ref, "kind": "metro", "line": ref, "colour": ln["colour"],
                    "path": [[round(x, 5), round(y, 5)] for x, y in g.coords],
                    "length_m": round(ln["geom_m"].length), "speed_mps": vcfg["metro_speed_kmh"] / 3.6,
                    "n": vcfg["metro_trains_per_line"]})
    affected = []
    for f in seg_features:
        if f["properties"]["infra_type"] == "street":
            affected += [r for r in f["properties"]["lines"] if r not in affected]
    others = [r for r in sorted(routes, key=lambda x: (len(x), x)) if r not in affected]
    for ref in (affected + others)[: vcfg["max_bus_routes"]]:
        r = routes[ref]
        g = to_ll(r["geom_m"])
        out.append({"id": f"BUS-{ref}", "kind": "bus", "line": ref, "colour": r["colour"],
                    "path": [[round(x, 5), round(y, 5)] for x, y in g.coords],
                    "length_m": round(r["geom_m"].length), "speed_mps": vcfg["bus_speed_kmh"] / 3.6,
                    "n": vcfg["buses_per_route"]})
    (GEO_DIR / "vehicles.json").write_text(json.dumps(out), encoding="utf-8")


# --------------------------------------------------------------------------- main
def build_network(offline_streets: bool = False) -> dict:
    db.init_db()
    lines, stations = build_metro()
    routes, stops = build_bus()
    _write("metro_lines.geojson", [
        _feature(to_ll(ln["geom_m_simpl"]), {"ref": ref, "name": ln["name"], "colour": ln["colour"]})
        for ref, ln in sorted(lines.items(), key=lambda kv: (len(kv[0]), kv[0]))])
    _write("metro_stations.geojson", [
        _feature(Point(sum(p[0] for p in st["pts"]) / len(st["pts"]), sum(p[1] for p in st["pts"]) / len(st["pts"])),
                 {"name": st["name"], "lines": st["lines"]}) for st in stations.values()])
    _write("bus_routes.geojson", [
        _feature(to_ll(r["geom_m"]), {"ref": ref, "name": r["name"], "colour": r["colour"]})
        for ref, r in sorted(routes.items(), key=lambda kv: (len(kv[0]), kv[0]))])
    _write("bus_stops.geojson", [
        _feature(Point(s["lon"], s["lat"]), {"name": s["name"], "lines": sorted(s["lines"])})
        for s in stops.values()])
    segs = build_segments(lines, stations, routes, stops, offline_streets)
    build_vehicle_paths(lines, routes, segs)
    cells = grid_cells()
    _write("weather_grid.geojson", [
        {"type": "Feature", "properties": {"cell_id": c.cell_id, "lon": c.lon, "lat": c.lat},
         "geometry": {"type": "Polygon", "coordinates": [[
             [round(c.lon_min, 5), round(c.lat_min, 5)], [round(c.lon_max, 5), round(c.lat_min, 5)],
             [round(c.lon_max, 5), round(c.lat_max, 5)], [round(c.lon_min, 5), round(c.lat_max, 5)],
             [round(c.lon_min, 5), round(c.lat_min, 5)]]]}} for c in cells.itertuples()])
    summary = {"metro_lines": len(lines), "metro_stations": len(stations), "bus_routes": len(routes),
               "bus_stops": len(stops), "segments": len(segs)}
    log.info("network built: %s", summary)
    return summary
