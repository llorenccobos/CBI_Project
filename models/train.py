"""Train the multi-horizon probabilistic water-level model.

For every horizon h in forecast.horizons_h and every target
  y_lvl : level at t+h
  y_max : maximum level over (t, t+h]
a LightGBM quantile regressor is trained for each quantile in
forecast.quantiles (one global model across all sensors).

Chronological split (config.model):  train | gap | validation | gap | calibration | gap | test
  * validation  -> early stopping
  * calibration -> conformal offsets, isotonic calibration of probabilities,
                   alarm threshold (target recall for threshold exceedance)
  * test        -> untouched, used only by models/evaluate.py
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import ARTIFACTS_DIR, LEVEL_ORDER, alert_level, settings, thresholds_for
from data_processing.features import ALL_FEATURES, FeatureStore
from models import uncertainty as U

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def resolve_splits(first: pd.Timestamp, last: pd.Timestamp, mcfg: dict) -> dict:
    """Split boundaries. Uses the configured dates when the data covers them
    (with >= 90 days of test data); otherwise proportional chronological
    splits (60 % train, 12 % validation, 12 % calibration, 16 % test)."""
    tr_end = pd.Timestamp(mcfg["train_end"]) + pd.Timedelta(hours=23)
    va_end = pd.Timestamp(mcfg["val_end"]) + pd.Timedelta(hours=23)
    ca_end = pd.Timestamp(mcfg["cal_end"]) + pd.Timedelta(hours=23)
    mode = "configured"
    if last - ca_end < pd.Timedelta(days=90) or tr_end <= first:
        span = last - first
        tr_end, va_end, ca_end = (first + span * 0.60).floor("D"), (first + span * 0.72).floor("D"), \
            (first + span * 0.84).floor("D")
        mode = "proportional (data does not cover configured dates)"
    return {"train_end": str(tr_end), "val_end": str(va_end), "cal_end": str(ca_end),
            "gap_hours": mcfg["gap_hours"], "mode": mode}


def split_masks(ts: pd.Series, sp: dict) -> dict:
    gap = pd.Timedelta(hours=sp["gap_hours"])
    tr_end, va_end, ca_end = (pd.Timestamp(sp[k]) for k in ("train_end", "val_end", "cal_end"))
    ts = pd.to_datetime(ts)
    return {
        "train": ts <= tr_end,
        "val": (ts > tr_end + gap) & (ts <= va_end),
        "cal": (ts > va_end + gap) & (ts <= ca_end),
        "test": ts > ca_end + gap,
    }


def eventful(df: pd.DataFrame) -> np.ndarray:
    return ((np.nan_to_num(df["y_max"].to_numpy()) > 3) | (np.nan_to_num(df["lvl_max24"].to_numpy()) > 3) |
            (np.nan_to_num(df["fc_sum"].to_numpy()) > 1.0) | (np.nan_to_num(df["rain_24h"].to_numpy()) > 2))


def downsample(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    ev = eventful(df)
    rng = np.random.default_rng(seed)
    keep = ev | (rng.random(len(df)) < frac)
    w = np.where(ev, 1.0, 1.0 / frac)[keep]
    return df[keep], w


def thresholds_array(df: pd.DataFrame, sensors: pd.DataFrame, level: str) -> np.ndarray:
    m = {sid: thresholds_for(sid, s.infra_type)[level] for sid, s in sensors.iterrows()}
    return df["sensor_id"].map(m).to_numpy(dtype=float)


def choose_alarm_threshold(p: np.ndarray, events: np.ndarray, target_recall: float) -> tuple[float, dict]:
    ok = np.isfinite(p)
    p, events = p[ok], events[ok].astype(bool)
    if events.sum() < 5:
        return 0.25, {"note": "too few events in calibration period; default 0.25", "n_events": int(events.sum())}
    best = 0.02
    for c in np.round(np.arange(0.02, 0.91, 0.01), 2):
        rec = (p[events] >= c).mean()
        if rec >= target_recall:
            best = c
    pred = p >= best
    return float(best), {"recall": float(pred[events].mean()), "precision": float(events[pred].mean()) if pred.any() else None,
                         "n_events": int(events.sum())}


def make_model(tau: float, params: dict) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="quantile", alpha=tau, n_jobs=-1, verbose=-1, random_state=7, **params)


def train(store: FeatureStore | None = None, horizons: list[int] | None = None) -> str:
    cfg = settings()
    mcfg, fcfg = cfg["model"], cfg["forecast"]
    taus = fcfg["quantiles"]
    horizons = horizons or fcfg["horizons_h"]
    t_start = time.time()
    store = store or FeatureStore()
    splits = resolve_splits(store.index[0], store.index[-1], mcfg)
    log.info("splits: %s", splits)
    stats = store.compute_sensor_stats(splits["train_end"])
    store.sensor_stats = stats
    base_cache = {sid: store.base_features(sid) for sid in store.sensors.index}
    version = time.strftime("%Y%m%d-%H%M%S")
    out = ARTIFACTS_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    params = dict(mcfg["lightgbm"])
    calibration, importances, info = {}, {}, {}
    alert = alert_level()
    for h in horizons:
        df = store.horizon_frame(h, base_cache=base_cache)
        masks = split_masks(df.ts, splits)
        tr, w_tr = downsample(df[masks["train"]], mcfg["dry_keep_fraction"], seed=h)
        va, w_va = downsample(df[masks["val"]], mcfg["dry_keep_fraction"], seed=100 + h)
        ca = df[masks["cal"]]
        info[h] = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_cal": int(len(ca))}
        for target in ("y_lvl", "y_max"):
            if h == 1 and target == "y_max":
                calibration[f"h{h}_{target}"] = calibration[f"h{h}_y_lvl"]
                iso_src = out / f"iso_h{h}_y_lvl.joblib"
                if iso_src.exists():
                    joblib.dump(joblib.load(iso_src), out / f"iso_h{h}_{target}.joblib")
                continue
            mtr, mva = tr[target].notna().to_numpy(), va[target].notna().to_numpy()
            Xtr, ytr = tr.loc[mtr, ALL_FEATURES], tr.loc[mtr, target]
            Xva, yva = va.loc[mva, ALL_FEATURES], va.loc[mva, target]
            boosters = []
            for tau in taus:
                m = make_model(tau, params)
                m.fit(Xtr, ytr, sample_weight=w_tr[mtr], eval_set=[(Xva, yva)], eval_sample_weight=[w_va[mva]],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
                m.booster_.save_model(str(out / f"h{h}_{target}_q{int(round(tau * 100)):02d}.txt"),
                                      num_iteration=m.best_iteration_)
                boosters.append(m.booster_)
                if tau == 0.5:
                    importances[f"h{h}_{target}"] = dict(sorted(
                        zip(ALL_FEATURES, map(float, m.booster_.feature_importance("gain"))),
                        key=lambda kv: -kv[1])[:20])
            # ---------------- calibration period
            mca = ca[target].notna().to_numpy()
            Xc, yc = ca.loc[mca, ALL_FEATURES], ca.loc[mca, target].to_numpy()
            Qc = U.rearrange(np.column_stack([b.predict(Xc) for b in boosters]))
            reg = U.regime(ca.loc[mca])
            offs = U.conformal_offsets(Qc, yc, reg, taus, tuple(fcfg["interval"]))
            Qc2 = U.apply_conformal(Qc, reg, offs, taus)
            p_all, e_all = [], []
            for lv in LEVEL_ORDER:
                thr = thresholds_array(ca.loc[mca], store.sensors, lv)
                p_all.append(U.prob_exceed(Qc2, taus, thr))
                e_all.append((yc > thr).astype(float))
            iso = U.fit_isotonic(np.concatenate(p_all), np.concatenate(e_all))
            if iso is not None:
                joblib.dump(iso, out / f"iso_h{h}_{target}.joblib")
            entry = {"conformal": {str(k): v for k, v in offs.items()}}
            if target == "y_max" or h == 1:
                thr_c = thresholds_array(ca.loc[mca], store.sensors, alert)
                p_cal = U.calibrate(iso, U.prob_exceed(Qc2, taus, thr_c))
                alarm_p, alarm_info = choose_alarm_threshold(p_cal, yc > thr_c, mcfg["target_recall"])
                entry.update({"alarm_p": alarm_p, "alarm_info": alarm_info})
            calibration[f"h{h}_{target}"] = entry
            log.info("h=%2d %s trained (n=%d) conformal=%s %s", h, target, len(Xtr), offs,
                     entry.get("alarm_info", ""))
    meta = {
        "version": version, "created": pd.Timestamp.now("UTC").isoformat(), "quantiles": taus,
        "horizons": horizons, "features": ALL_FEATURES, "sensor_stats": stats, "splits": splits,
        "data": {"first_ts": str(store.index[0]), "last_ts": str(store.index[-1]),
                 "sensors": list(store.sensors.index), "synthetic_sensors": True},
        "rows": info, "lightgbm": params, "train_seconds": round(time.time() - t_start, 1),
        "importance_gain": importances,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "calibration.json").write_text(json.dumps(calibration, indent=1))
    (ARTIFACTS_DIR / "latest.txt").write_text(version)
    log.info("model %s trained in %.0f s", version, time.time() - t_start)
    return version
