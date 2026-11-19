"""SYNTHETIC water-level sensors (clearly flagged `is_synthetic = 1`).

No public water-level data exists for Barcelona streets or metro sections, so
this module simulates them from REAL hourly rainfall (Open-Meteo grid cell of
each sensor + its upstream cell) with a simple conceptual hydrological model:

  street : rain -> runoff (wetness-dependent coefficient) -> surface storage
           drained by the sewer inlets at a limited capacity -> ponding depth
  metro  : surface excess entering the tunnel (entrances / vents) + slow
           groundwater seepage -> tunnel storage emptied by pumps -> depth

On top of the "true" level, realistic sensor artefacts are added: noise,
communication gaps, spikes and stuck readings. They are later detected by the
quality-control step in data_processing.cleaning.

The generator is deterministic (seeded) and prefix-stable: extending the
weather history to "now" reproduces exactly the same past values.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from config.settings import settings
from data_storage import db
from data_storage.repository import load_sensors, load_weather

log = logging.getLogger(__name__)

# level reached by the 99.95th-percentile storm hour, per vulnerability
TARGET_PEAK_CM = {"high": 42.0, "medium": 31.0, "low": 20.0}
# physical ceiling: beyond this depth water spills to neighbouring streets / sections
MAX_DEPTH_CM = {"street": 75.0, "metro": 85.0}


def _rng(sensor_id: str, stream: str) -> np.random.Generator:
    seed = settings()["sensors"]["synthetic_seed"]
    h = int(hashlib.sha256(f"{seed}-{sensor_id}-{stream}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def sensor_params(sensor_id: str, infra_type: str, vulnerability: str) -> dict:
    r = _rng(sensor_id, "params")
    v = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(vulnerability, 0.5)
    p = {
        "catchment": r.uniform(0.8, 1.6) * (1 + 0.6 * v),     # relative contributing area
        "c0": r.uniform(0.35, 0.55),                          # dry runoff coefficient
        "upstream_share": r.uniform(0.2, 0.6),                # runoff arriving from uphill
        "drain_capacity": r.uniform(6, 12) * (1.3 - 0.5 * v),  # mm/h evacuated by inlets / pumps
        "decay": r.uniform(0.04, 0.10),                       # infiltration / losses per hour
    }
    if infra_type == "metro":
        p.update({
            "entry_share": r.uniform(0.25, 0.5),               # fraction of surface excess entering
            "lag": r.uniform(0.6, 0.8),                        # tunnel inflow smoothing (0..1)
            "seepage": r.uniform(0.004, 0.012) * (1 + v),      # groundwater contribution
            "base_cm": r.uniform(0.3, 1.5),                    # permanent seepage film
        })
    return p


def simulate_true_level(P: np.ndarray, Pu: np.ndarray, infra_type: str, p: dict) -> np.ndarray:
    n = len(P)
    S = 0.0; api = 0.0; T = 0.0; G = 0.0; inflow_t = 0.0
    out = np.zeros(n)
    Pu_lag = np.concatenate([[0.0], Pu[:-1]])
    for t in range(n):
        rain = P[t] if np.isfinite(P[t]) else 0.0
        up = Pu_lag[t] if np.isfinite(Pu_lag[t]) else 0.0
        api = api * 0.97 + rain
        c = p["c0"] + (1 - p["c0"]) * min(1.0, api / 60.0)
        inflow = p["catchment"] * c * (rain + p["upstream_share"] * up)
        S += inflow
        drained = min(0.35 * S, p["drain_capacity"])   # linear-reservoir recession, capped by inlet capacity
        S = max(0.0, (S - drained) * (1 - p["decay"]))
        if infra_type == "street":
            out[t] = S
        else:
            excess = max(0.0, inflow - 0.5 * p["drain_capacity"])
            inflow_t = p["lag"] * inflow_t + (1 - p["lag"]) * excess * p["entry_share"]
            G = G * 0.996 + 0.02 * rain
            T += inflow_t + p["seepage"] * G
            pumped = min(0.22 * T, p["drain_capacity"] * 0.3)       # pumping capacity limits the recession
            T = max(0.0, T - pumped)
            out[t] = T
    return out


def generate(write: bool = True) -> pd.DataFrame:
    sensors = load_sensors()
    if sensors.empty:
        raise RuntimeError("No sensors in database: run the network build first.")
    w = load_weather()[["cell_id", "ts", "precipitation"]]
    pr = w.pivot_table(index="ts", columns="cell_id", values="precipitation", aggfunc="mean").sort_index()
    pr = pr.asfreq("h")
    pr = pr.loc[pr.first_valid_index(): pr.last_valid_index()]
    idx = pr.index
    n = len(idx)
    frames = []
    for s in sensors.itertuples():
        p = sensor_params(s.sensor_id, s.infra_type, s.vulnerability)
        P = pr[s.weather_cell].to_numpy(dtype=float)
        Pu = pr[s.upstream_cell].to_numpy(dtype=float)
        S = simulate_true_level(P, Pu, s.infra_type, p)
        # logarithmic storage -> depth curve (gradual rise / recession), capped at the
        # spill depth; calibrated on a fixed 3-year prefix so the series is prefix-stable
        ref = max(float(np.nanquantile(S[: min(n, 3 * 8760)], 0.9995)), 0.05)
        cap = MAX_DEPTH_CM[s.infra_type]
        target = TARGET_PEAK_CM.get(s.vulnerability, 30.0)
        b = ref
        a = target / np.log(2.0)
        depth = a * np.log1p(S / b)
        true = cap * np.tanh(depth / cap) + p.get("base_cm", 0.0)   # smooth approach to the cap
        # ---------------- sensor artefacts (prefix-stable random streams)
        r_noise = _rng(s.sensor_id, "noise").normal(size=n + 200000)[:n]
        r_evt = _rng(s.sensor_id, "events").uniform(size=(n + 200000, 4))[:n]
        meas = true + np.where(true > 0.5, 0.35, 0.12) * r_noise
        meas = np.abs(meas) if s.infra_type == "street" else np.maximum(meas, 0)
        quality = np.full(n, "good", dtype=object)
        # communication gaps: start prob 0.0008/h, mean duration ~6 h
        gap_left = 0
        stuck_left = 0
        stuck_val = 0.0
        for t in range(n):
            if gap_left > 0:
                meas[t] = np.nan; quality[t] = "missing"; gap_left -= 1; continue
            if stuck_left > 0:
                meas[t] = stuck_val; stuck_left -= 1; continue
            u = r_evt[t]
            if u[0] < 0.0008:
                gap_left = int(1 + u[1] * 12); meas[t] = np.nan; quality[t] = "missing"
            elif u[0] < 0.0008 + 0.0004:
                meas[t] += 6 + 18 * u[1]                     # spike (debris / electrical)
            elif u[0] < 0.0008 + 0.0004 + 0.00015 and true[t] > 1:
                stuck_left = int(6 + u[2] * 24); stuck_val = meas[t]  # frozen reading
        frames.append(pd.DataFrame({"sensor_id": s.sensor_id, "ts": idx, "level_cm": np.round(meas, 2),
                                    "quality": quality, "is_synthetic": True}))
        log.info("synthetic %s: max %.1f cm, hours >= 30 cm: %d", s.sensor_id, np.nanmax(true),
                 int((true >= 30).sum()))
    df = pd.concat(frames, ignore_index=True)
    if write:
        db.upsert_df(db.observations, df)
    return df
