"""Evaluation on the untouched TEST period (chronologically after calibration).

Reports, per horizon:
  * MAE / RMSE of the median level (all hours and "wet" hours)
  * error of the predicted window maximum on eventful windows
  * 90 % interval coverage (PICP) and mean width
  * threshold exceedance (alert level, window max): precision, recall,
    false-negative rate, false-alarm ratio, Brier score, reliability curve
  * event-level detection: share of flood events (onsets) announced h hours ahead
Compared against baselines:
  * persistence       : level(t+h) = level(t)
  * climatology       : per-sensor mean level by hour of day (training period)
  * rain-linear       : ridge regression on rain / level features
Model comparison (median level, subset of horizons): LightGBM vs XGBoost vs
CatBoost vs Random Forest, plus an ablation without spatial features.
Outputs: reports/evaluation.json and reports/model_report.md
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from config.settings import REPORTS_DIR, alert_level, settings
from data_processing.features import ALL_FEATURES, SPATIAL_FEATURES, FeatureStore
from models import uncertainty as U
from models.predict import ModelBundle, latest_version
from models.train import downsample, split_masks, thresholds_array

log = logging.getLogger(__name__)


def _reg_metrics(y, yhat, wet=None) -> dict:
    m = np.isfinite(y) & np.isfinite(yhat)
    e = yhat[m] - y[m]
    out = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "n": int(m.sum())}
    if wet is not None:
        mw = m & wet
        if mw.sum():
            ew = yhat[mw] - y[mw]
            out.update({"mae_wet": float(np.mean(np.abs(ew))), "rmse_wet": float(np.sqrt(np.mean(ew ** 2))),
                        "n_wet": int(mw.sum())})
    return out


def _cls_metrics(p, event, cut) -> dict:
    m = np.isfinite(p)
    p, ev = p[m], event[m].astype(bool)
    pred = p >= cut
    tp, fp, fn = int((pred & ev).sum()), int((pred & ~ev).sum()), int((~pred & ev).sum())
    out = {"events": int(ev.sum()), "alarms": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
           "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
           "fnr": fn / (tp + fn) if tp + fn else None, "far": fp / (tp + fp) if tp + fp else None,
           "brier": float(np.mean((p - ev) ** 2))}
    bins = np.linspace(0, 1, 11)
    rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        mm = (p >= a) & (p < b if b < 1 else p <= b)
        if mm.sum() >= 5:
            rel.append({"bin": [round(a, 1), round(b, 1)], "mean_p": float(p[mm].mean()),
                        "freq": float(ev[mm].mean()), "n": int(mm.sum())})
    out["reliability"] = rel
    return out


def _events(series: pd.Series, thr: float, merge_gap: int = 3) -> list[pd.Timestamp]:
    above = (series >= thr).fillna(False).to_numpy()
    onsets, last_end = [], -10 ** 9
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            if i - last_end > merge_gap:
                onsets.append(series.index[i])
            last_end = j
            i = j + 1
        else:
            i += 1
    return onsets


def evaluate(compare_models: bool = True) -> dict:
    cfg = settings()
    mcfg = cfg["model"]
    version = latest_version()
    bundle = ModelBundle(version)
    store = FeatureStore(sensor_stats=bundle.sensor_stats)
    base_cache = {sid: store.base_features(sid) for sid in store.sensors.index}
    taus = bundle.taus
    i_lo, i_md, i_hi = taus.index(0.05), taus.index(0.5), taus.index(0.95)
    alert = alert_level()
    # climatology baseline: per sensor x hour-of-day mean over the training period
    train_part = store.clean.loc[:bundle.meta["splits"]["train_end"]]
    clim = train_part.groupby(train_part.index.hour).mean()
    results = {"model_version": version, "test_start": None, "horizons": {}, "events": {}, "comparison": {}}
    event_probs = {}
    for h in bundle.horizons:
        df = store.horizon_frame(h, base_cache=base_cache)
        masks = split_masks(df.ts, bundle.meta["splits"])
        te = df[masks["test"]].reset_index(drop=True)
        tr_full = df[masks["train"]]
        if te.empty:
            log.warning("empty test set for h=%d", h)
            continue
        results["test_start"] = str(te.ts.min())
        results["test_end"] = str(te.ts.max())
        y, ymax = te.y_lvl.to_numpy(), te.y_max.to_numpy()
        wet = (np.nan_to_num(y) > 1) | (np.nan_to_num(te.lvl_0.to_numpy()) > 1)
        Ql = bundle.quantiles(te, h, "y_lvl")
        Qm = Ql if h == 1 else bundle.quantiles(te, h, "y_max")
        thr = thresholds_array(te, store.sensors, alert)
        p_max = bundle.exceedance(Qm, thr, h, "y_max")
        cut = bundle.alarm_threshold(h)
        # baselines
        pers = te.lvl_0.to_numpy()
        th = (pd.to_datetime(te.ts) + pd.Timedelta(hours=h)).dt.hour.to_numpy()
        climv = np.array([clim.at[hh, sid] if sid in clim.columns else np.nan for hh, sid in zip(th, te.sensor_id)])
        trs, _ = downsample(tr_full[tr_full.y_lvl.notna()], 0.1, seed=5)
        lin_cols = ["lvl_0", "lvl_d1", "rain_1h", "rain_3h", "rain_6h", "fc_sum", "fc_max", "fc_last3", "api", "fcu_sum"]
        ridge = Ridge(alpha=1.0).fit(trs[lin_cols].fillna(0), trs.y_lvl)
        lin = np.clip(ridge.predict(te[lin_cols].fillna(0)), 0, None)
        ev_max = np.nan_to_num(ymax) > thr
        eventful_win = np.nan_to_num(ymax) >= 10
        r = {
            "lightgbm_quantile": _reg_metrics(y, Ql[:, i_md], wet),
            "persistence": _reg_metrics(y, pers, wet),
            "climatology": _reg_metrics(y, climv, wet),
            "rain_linear": _reg_metrics(y, lin, wet),
            "max_level_error": {
                "lightgbm": _reg_metrics(ymax[eventful_win], Qm[eventful_win, i_md]),
                "persistence": _reg_metrics(ymax[eventful_win], pers[eventful_win]),
            },
            "interval": {
                "coverage90": float(np.nanmean((y >= Ql[:, i_lo]) & (y <= Ql[:, i_hi]))),
                "coverage90_wet": float(np.nanmean(((y >= Ql[:, i_lo]) & (y <= Ql[:, i_hi]))[wet])) if wet.any() else None,
                "mean_width": float(np.mean(Ql[:, i_hi] - Ql[:, i_lo])),
                "mean_width_wet": float(np.mean((Ql[:, i_hi] - Ql[:, i_lo])[wet])) if wet.any() else None,
            },
            "exceedance_max": _cls_metrics(p_max, ev_max, cut) | {"alarm_p": cut},
            "exceedance_persistence": _cls_metrics((pers >= thr).astype(float), ev_max, 0.5),
        }
        results["horizons"][str(h)] = r
        event_probs[h] = te.assign(p=p_max)[["sensor_id", "ts", "p"]].set_index(["sensor_id", "ts"])["p"]
        log.info("h=%2d MAE %.2f (pers %.2f) recall %s precision %s cov90 %.2f", h, r["lightgbm_quantile"]["mae"],
                 r["persistence"]["mae"], r["exceedance_max"]["recall"], r["exceedance_max"]["precision"],
                 r["interval"]["coverage90"])
    # ------------------------------------------------ event-level detection
    test_start = pd.Timestamp(results["test_start"]) if results["test_start"] else None
    det = {str(h): {"detected": 0, "total": 0} for h in bundle.horizons}
    n_events = 0
    if test_start is not None:
        for sid in store.sensors.index:
            thr_s = thresholds_array(pd.DataFrame({"sensor_id": [sid]}), store.sensors, alert)[0]
            onsets = [o for o in _events(store.clean[sid], thr_s) if o >= test_start + pd.Timedelta(hours=72)]
            n_events += len(onsets)
            for o in onsets:
                for h in bundle.horizons:
                    key = (sid, o - pd.Timedelta(hours=h))
                    if h in event_probs and key in event_probs[h].index:
                        det[str(h)]["total"] += 1
                        det[str(h)]["detected"] += int(event_probs[h].loc[key] >= bundle.alarm_threshold(h))
    results["events"] = {"n_events": n_events, "detection_by_lead": det}
    # ------------------------------------------------ model comparison
    if compare_models:
        results["comparison"] = compare(store, base_cache, bundle)
    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(results, bundle)
    return results


def compare(store, base_cache, bundle, horizons=(1, 3, 6, 24)) -> dict:
    """Median-level comparison of tabular learners on identical splits."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    mcfg = settings()["model"]
    splits = bundle.meta["splits"]
    out = {}
    for h in horizons:
        if h not in bundle.horizons:
            continue
        df = store.horizon_frame(h, base_cache=base_cache)
        masks = split_masks(df.ts, bundle.meta["splits"])
        tr, w = downsample(df[masks["train"] & df.y_lvl.notna()], mcfg["dry_keep_fraction"], seed=11)
        va, wv = downsample(df[masks["val"] & df.y_lvl.notna()], mcfg["dry_keep_fraction"], seed=12)
        te = df[masks["test"] & df.y_lvl.notna()]
        wet = (te.y_lvl.to_numpy() > 1) | (np.nan_to_num(te.lvl_0.to_numpy()) > 1)
        res = {}
        no_sp = [f for f in ALL_FEATURES if f not in SPATIAL_FEATURES]
        specs = {
            "LightGBM": lambda: lgb.LGBMRegressor(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=31,
                                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1),
            "XGBoost": lambda: xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=600, learning_rate=0.05,
                                                max_depth=7, subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                                early_stopping_rounds=50),
            "CatBoost": lambda: CatBoostRegressor(loss_function="MAE", iterations=600, learning_rate=0.08, depth=7,
                                                  verbose=0, thread_count=-1, early_stopping_rounds=50),
            "RandomForest": lambda: RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                                          n_jobs=-1, random_state=0),
        }
        for name, mk in specs.items():
            t0 = time.time()
            m = mk()
            Xtr, Xva, Xte = tr[ALL_FEATURES], va[ALL_FEATURES], te[ALL_FEATURES]
            if name == "LightGBM":
                m.fit(Xtr, tr.y_lvl, sample_weight=w, eval_set=[(Xva, va.y_lvl)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            elif name in ("XGBoost", "CatBoost"):
                kw = {"verbose": False}
                m.fit(Xtr, tr.y_lvl, sample_weight=w, eval_set=[(Xva, va.y_lvl)], **kw)
            else:
                m.fit(Xtr.fillna(-1), tr.y_lvl, sample_weight=w)
                Xte = Xte.fillna(-1)
            pred = np.clip(m.predict(Xte), 0, None)
            res[name] = _reg_metrics(te.y_lvl.to_numpy(), pred, wet) | {"fit_s": round(time.time() - t0, 1)}
        m = specs["LightGBM"]()
        m.fit(tr[no_sp], tr.y_lvl, sample_weight=w, eval_set=[(va[no_sp], va.y_lvl)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        res["LightGBM (no spatial features)"] = _reg_metrics(te.y_lvl.to_numpy(),
                                                             np.clip(m.predict(te[no_sp]), 0, None), wet)
        out[str(h)] = res
        log.info("comparison h=%d: %s", h, {k: round(v["mae_wet"], 2) for k, v in res.items() if "mae_wet" in v})
    return out


def _f(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(r: dict, bundle) -> None:
    L = [f"# Model evaluation report", "",
         f"Model version **{r['model_version']}** · test period {r.get('test_start')} → {r.get('test_end')}", "",
         "> Sensor data are **SYNTHETIC** (physically-based simulator driven by real Open-Meteo rainfall).",
         "> Metrics show that the pipeline works end-to-end; they are not evidence of real-world skill.", "",
         "## Water level (median forecast) — MAE / RMSE in cm", "",
         "| Horizon | LightGBM MAE | wet MAE | Persistence wet MAE | Climatology wet MAE | Rain-linear wet MAE | 90% coverage | 90% coverage (wet) |",
         "|---|---|---|---|---|---|---|---|"]
    for h, m in r["horizons"].items():
        L.append(f"| +{h} h | {_f(m['lightgbm_quantile']['mae'])} | {_f(m['lightgbm_quantile'].get('mae_wet'))} | "
                 f"{_f(m['persistence'].get('mae_wet'))} | {_f(m['climatology'].get('mae_wet'))} | "
                 f"{_f(m['rain_linear'].get('mae_wet'))} | {_f(m['interval']['coverage90'])} | "
                 f"{_f(m['interval']['coverage90_wet'])} |")
    L += ["", "## Threshold exceedance (window maximum ≥ alert threshold)", "",
          "| Horizon | Alarm p* | Events | Recall | FNR | Precision | False-alarm ratio | Brier | Persistence recall |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h, m in r["horizons"].items():
        e, pe = m["exceedance_max"], m["exceedance_persistence"]
        L.append(f"| +{h} h | {_f(e['alarm_p'])} | {e['events']} | {_f(e['recall'])} | {_f(e['fnr'])} | "
                 f"{_f(e['precision'])} | {_f(e['far'])} | {_f(e['brier'], 4)} | {_f(pe['recall'])} |")
    L += ["", "## Window-maximum error on eventful windows (max ≥ 10 cm)", "",
          "| Horizon | LightGBM MAE | Persistence MAE |", "|---|---|---|"]
    for h, m in r["horizons"].items():
        L.append(f"| +{h} h | {_f(m['max_level_error']['lightgbm'].get('mae'))} | "
                 f"{_f(m['max_level_error']['persistence'].get('mae'))} |")
    ev = r.get("events", {})
    L += ["", f"## Event-level detection ({ev.get('n_events', 0)} flood onsets in test period)", "",
          "| Lead time | Detected / total |", "|---|---|"]
    for h, d in ev.get("detection_by_lead", {}).items():
        L.append(f"| {h} h ahead | {d['detected']} / {d['total']} |")
    if r.get("comparison"):
        L += ["", "## Learner comparison (median level, wet-hours MAE in cm)", ""]
        names = list(next(iter(r["comparison"].values())).keys())
        L += ["| Horizon | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for h, res in r["comparison"].items():
            L.append(f"| +{h} h | " + " | ".join(_f(res[n].get("mae_wet")) for n in names) + " |")
    L += ["", "## Most important features (gain, median model)", ""]
    for key in ("h1_y_lvl", "h6_y_max", "h24_y_max"):
        imp = bundle.meta.get("importance_gain", {}).get(key)
        if imp:
            L.append(f"* **{key}**: " + ", ".join(list(imp)[:8]))
    (REPORTS_DIR / "model_report.md").write_text("\n".join(L), encoding="utf-8")
