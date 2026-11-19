"""Inference: load trained artifacts and produce probabilistic forecasts.

`ModelBundle`  loads the boosters + calibration objects of one model version
               and turns a feature frame into calibrated quantiles and
               threshold-exceedance probabilities.
`Forecaster`   builds the features for an issue time (live or replay of a
               past event) and returns a JSON-ready forecast for every sensor.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import (ARTIFACTS_DIR, LEVEL_ORDER, alert_level, risk_class, settings,
                             thresholds_for)
from data_processing.features import FEATURE_TO_GROUP, GROUP_LABELS, LMAX, FeatureStore
from data_storage.repository import last_observation_time, latest_forecast_run
from models import uncertainty as U

log = logging.getLogger(__name__)


class ModelNotTrained(RuntimeError):
    pass


def latest_version() -> str | None:
    f = ARTIFACTS_DIR / "latest.txt"
    return f.read_text().strip() if f.exists() else None


class ModelBundle:
    def __init__(self, version: str):
        self.version = version
        self.dir: Path = ARTIFACTS_DIR / version
        if not self.dir.exists():
            raise ModelNotTrained(f"model version {version} not found")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.taus = self.meta["quantiles"]
        self.features = self.meta["features"]
        self.horizons = self.meta["horizons"]
        self.sensor_stats = self.meta["sensor_stats"]
        self.calib = json.loads((self.dir / "calibration.json").read_text())
        self._boosters: dict = {}
        self._iso: dict = {}

    @classmethod
    def load_latest(cls) -> "ModelBundle":
        v = latest_version()
        if not v:
            raise ModelNotTrained("No trained model found. Run: python main.py train")
        return cls(v)

    def boosters(self, h: int, target: str) -> list[lgb.Booster]:
        key = (h, target)
        if key not in self._boosters:
            name_t = "y_lvl" if (h == 1 and target == "y_max") else target
            self._boosters[key] = [lgb.Booster(model_file=str(self.dir / f"h{h}_{name_t}_q{int(round(t * 100)):02d}.txt"))
                                   for t in self.taus]
        return self._boosters[key]

    def iso(self, h: int, target: str):
        key = (h, target)
        if key not in self._iso:
            f = self.dir / f"iso_h{h}_{target}.joblib"
            self._iso[key] = joblib.load(f) if f.exists() else None
        return self._iso[key]

    def raw_quantiles(self, X: pd.DataFrame, h: int, target: str) -> np.ndarray:
        Q = np.column_stack([b.predict(X[self.features]) for b in self.boosters(h, target)])
        return U.rearrange(Q)

    def quantiles(self, X: pd.DataFrame, h: int, target: str) -> np.ndarray:
        Q = self.raw_quantiles(X, h, target)
        offs = {int(k): v for k, v in self.calib[f"h{h}_{target}"]["conformal"].items()}
        return U.apply_conformal(Q, U.regime(X), offs, self.taus)

    def exceedance(self, Q: np.ndarray, thr, h: int, target: str) -> np.ndarray:
        return U.calibrate(self.iso(h, target), U.prob_exceed(Q, self.taus, thr))

    def alarm_threshold(self, h: int) -> float:
        return float(self.calib[f"h{h}_y_max"].get("alarm_p", 0.3))

    def contributions(self, X: pd.DataFrame, h: int) -> list[list[dict]]:
        """Grouped feature contributions (cm) of the median level model, per row."""
        b = self.boosters(h, "y_lvl")[self.taus.index(0.5)]
        C = b.predict(X[self.features], pred_contrib=True)
        out = []
        for row in C:
            groups: dict = {}
            for f, v in zip(self.features, row[:-1]):
                g = FEATURE_TO_GROUP.get(f, "other")
                groups[g] = groups.get(g, 0.0) + float(v)
            items = sorted(groups.items(), key=lambda kv: -abs(kv[1]))
            out.append([{"group": g, "label": GROUP_LABELS.get(g, g), "cm": round(v, 2)} for g, v in items[:5]])
        return out


@lru_cache(maxsize=1)
def _bundle_cached(version: str) -> ModelBundle:
    return ModelBundle(version)


def get_bundle() -> ModelBundle:
    v = latest_version()
    if not v:
        raise ModelNotTrained("No trained model found. Run: python main.py train")
    return _bundle_cached(v)


# =========================================================================== forecaster
def _live_forecast_matrix(store: FeatureStore, t0: pd.Timestamp):
    """Forecast precipitation matrix {cell: (1, LMAX)} from the latest live run, if fresh."""
    issued, run = latest_forecast_run()
    if issued is None or run.empty or issued < t0 - pd.Timedelta(hours=6):
        return None, None
    out = {}
    valid = [t0 + pd.Timedelta(hours=L) for L in range(1, LMAX + 1)]
    obs = store.weather["precipitation"]
    for cell in store.cells:
        r = run[run.cell_id == cell].set_index("ts")["precipitation"]
        vals = []
        for v in valid:
            if v in r.index:
                vals.append(r.loc[v])
            elif v in obs.index and pd.notna(obs.at[v, cell]):   # hours between run issue and t0
                vals.append(obs.at[v, cell])
            else:
                vals.append(np.nan)
        out[cell] = np.asarray(vals, dtype=np.float32)[None, :]
    return out, issued


def forecast(issue_time=None) -> dict:
    """Probabilistic forecast for all sensors at `issue_time` (default: latest data)."""
    bundle = get_bundle()
    cfg = settings()
    last = last_observation_time()
    if last is None:
        raise RuntimeError("No sensor data available")
    t0 = pd.Timestamp(issue_time).floor("h") if issue_time is not None else last
    t0 = min(t0, last)
    live = issue_time is None or t0 >= last - pd.Timedelta(hours=1)
    store = FeatureStore(start=t0 - pd.Timedelta(days=10), end=t0, sensor_stats=bundle.sensor_stats)
    pos = np.array([len(store.index) - 1])
    t0 = store.index[-1]
    fc_override, fc_issued, fc_source = None, None, "archived-forecast-reconstruction"
    if live:
        fc_override, fc_issued = _live_forecast_matrix(store, t0)
        fc_source = "live-forecast" if fc_override is not None else "no-live-forecast (archived/persistence proxy)"
    coverage_h = LMAX
    if fc_override is not None:
        first = next(iter(fc_override.values()))[0]
        coverage_h = int(np.sum(~np.isnan(first)))
    weather_ok = (fc_override is not None and coverage_h >= 24) or not live
    base_cache = {sid: store.base_features(sid) for sid in store.sensors.index}
    sensors = list(store.sensors.index)
    per_h = {}
    for h in bundle.horizons:
        X = store.horizon_frame(h, positions=pos, forecast_override=fc_override, base_cache=base_cache)
        Ql = bundle.quantiles(X, h, "y_lvl")
        Qm = Ql if h == 1 else bundle.quantiles(X, h, "y_max")
        per_h[h] = (X, Ql, Qm, bundle.contributions(X, h))
    taus = bundle.taus
    i_lo, i_md, i_hi = taus.index(0.05), taus.index(0.5), taus.index(0.95)
    alert = alert_level()
    results = []
    for k, sid in enumerate(sensors):
        s = store.sensors.loc[sid]
        thr = thresholds_for(sid, s.infra_type)
        cur_raw = store.clean[sid].iloc[-1]
        cur = store.filled[sid].iloc[-1]
        stale = float(store.stale[sid].iloc[-1])
        flag = store.flags[sid].iloc[-1]
        hz = []
        first_exc = None
        for h in bundle.horizons:
            X, Ql, Qm, contrib = per_h[h]
            ql, qm = Ql[k:k + 1], Qm[k:k + 1]
            p_lvl = {lv: float(bundle.exceedance(ql, thr[lv], h, "y_lvl")[0]) for lv in LEVEL_ORDER}
            p_max = {lv: float(bundle.exceedance(qm, thr[lv], h, "y_max")[0]) for lv in LEVEL_ORDER}
            width = float(ql[0, i_hi] - ql[0, i_lo])
            score, label, reasons = U.confidence(h, width, thr[alert], stale, weather_ok,
                                                 cfg["forecast"]["operational_max_h"])
            alarm = p_max[alert] >= bundle.alarm_threshold(h)
            if alarm and first_exc is None:
                prev = [x for x in bundle.horizons if x < h]
                first_exc = {"within_h": h, "after_h": prev[-1] if prev else 0,
                             "probability": round(p_max[alert], 3)}
            hz.append({
                "h": h, "valid_time": (t0 + pd.Timedelta(hours=h)).isoformat(),
                "tier": "operational" if h <= cfg["forecast"]["operational_max_h"] else "awareness",
                "level": {"median": round(float(ql[0, i_md]), 2), "lo": round(float(ql[0, i_lo]), 2),
                          "hi": round(float(ql[0, i_hi]), 2),
                          "quantiles": {f"{t:.2f}": round(float(v), 2) for t, v in zip(taus, ql[0])}},
                "max": {"median": round(float(qm[0, i_md]), 2), "lo": round(float(qm[0, i_lo]), 2),
                        "hi": round(float(qm[0, i_hi]), 2)},
                "p_exceed": {k2: round(v, 4) for k2, v in p_lvl.items()},
                "p_exceed_max": {k2: round(v, 4) for k2, v in p_max.items()},
                "risk": risk_class(float(ql[0, i_md]), thr),
                "risk_max": risk_class(float(qm[0, i_md]), thr),
                "risk_upper": risk_class(float(ql[0, i_hi]), thr),
                "alarm": bool(alarm), "alarm_p": round(bundle.alarm_threshold(h), 3),
                "confidence": {"score": round(score, 2), "label": label, "reasons": reasons},
                "drivers": contrib[k],
                "inputs": {"fc_rain_window_mm": round(float(X["fc_sum"].iloc[k]), 2)
                           if pd.notna(X["fc_sum"].iloc[k]) else None,
                           "rain_6h_mm": round(float(X["rain_6h"].iloc[k]), 2)},
            })
        results.append({
            "sensor_id": sid, "name": s["name"], "infra_type": s.infra_type, "thresholds": thr,
            "current": {"ts": t0.isoformat(), "level": None if pd.isna(cur) else round(float(cur), 2),
                        "measured": None if pd.isna(cur_raw) else round(float(cur_raw), 2),
                        "quality": flag, "stale_h": stale,
                        "risk": risk_class(float(cur), thr) if pd.notna(cur) else "unknown"},
            "horizons": hz, "first_exceedance": first_exc,
        })
    return {
        "issued_at": t0.isoformat(), "mode": "live" if live else "replay", "model_version": bundle.version,
        "weather_forecast": {"source": fc_source, "issued_at": fc_issued.isoformat() if fc_issued is not None else None,
                             "ok": weather_ok, "coverage_h": coverage_h if fc_override is not None or not live else 0},
        "alert_level": alert, "horizons": bundle.horizons, "sensors": results,
        "synthetic_sensors": True,
    }
