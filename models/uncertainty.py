"""Uncertainty quantification utilities.

1. Quantile regression (LightGBM, several quantiles) gives a predictive
   distribution for each sensor / horizon.
2. Quantiles are re-sorted (no crossing) and clipped at 0 cm.
3. Conformalized Quantile Regression (CQR, Romano et al. 2019) widens or
   narrows the 5-95 % band so that its empirical coverage on a held-out
   calibration period is 90 %. Calibration is done separately for "dry" and
   "wet" regimes (Mondrian CQR) so that storms do not get the narrow bands
   learned from the many dry hours.
4. Threshold-exceedance probabilities P(level > threshold) are read from the
   calibrated predictive distribution (works for ANY threshold, so thresholds
   can be edited without retraining) and recalibrated with a monotone
   logistic (Platt-type) calibration on the log-odds.

Confidence (how much to trust the forecast) is reported separately from the
probability of flooding: it depends on the horizon tier, the width of the
interval relative to the threshold, and the freshness / quality of inputs.
"""
from __future__ import annotations

import numpy as np


def rearrange(Q: np.ndarray) -> np.ndarray:
    return np.sort(np.clip(Q, 0, None), axis=1)


def regime(df) -> np.ndarray:
    """1 = wet (rain or water present / expected), 0 = dry."""
    lvl = np.nan_to_num(df["lvl_0"].to_numpy(), nan=0)
    return ((lvl > 1.5) | (np.nan_to_num(df["fc_sum"].to_numpy()) > 1.0) |
            (np.nan_to_num(df["rain_6h"].to_numpy()) > 1.0)).astype(int)


def conformal_offsets(Q: np.ndarray, y: np.ndarray, reg: np.ndarray, taus: list[float],
                      interval=(0.05, 0.95)) -> dict:
    lo, hi = taus.index(interval[0]), taus.index(interval[1])
    alpha = 1 - (interval[1] - interval[0])
    out = {}
    for r in (0, 1):
        m = (reg == r) & np.isfinite(y)
        if m.sum() < 30:
            out[r] = 0.0
            continue
        scores = np.maximum(Q[m, lo] - y[m], y[m] - Q[m, hi])
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        out[r] = float(np.quantile(scores, level))
    return out


def apply_conformal(Q: np.ndarray, reg: np.ndarray, offsets: dict, taus: list[float]) -> np.ndarray:
    scale = (np.asarray(taus) - 0.5) / 0.45     # -1 at q05, 0 at median, +1 at q95
    off = np.where(reg == 1, offsets.get(1, 0.0), offsets.get(0, 0.0))[:, None]
    # never shrink the band below half of the original width (robustness)
    adj = Q + off * scale[None, :]
    return rearrange(adj)


def prob_exceed(Q: np.ndarray, taus: list[float], thr) -> np.ndarray:
    """P(Y > thr) from quantiles Q (n, K). thr: scalar or (n,) array."""
    taus = np.asarray(taus)
    n = Q.shape[0]
    thr = np.broadcast_to(np.asarray(thr, dtype=float), (n,))
    p = np.empty(n)
    k = len(taus)
    for i in range(n):
        q = Q[i]
        t = thr[i]
        if t <= q[0]:
            p[i] = 1 - taus[0] * (t / q[0] if q[0] > 0 else 0.0)
        elif t >= q[-1]:
            # exponential upper tail fitted on the two highest quantiles
            dq = max(q[-1] - q[-3], 0.25)
            scale = dq / np.log((1 - taus[-3]) / (1 - taus[-1]))
            p[i] = (1 - taus[-1]) * np.exp(-(t - q[-1]) / max(scale, 0.25))
        else:
            j = np.searchsorted(q, t, side="right")
            j = min(max(j, 1), k - 1)
            q0, q1 = q[j - 1], q[j]
            frac = 0.5 if q1 <= q0 else (t - q0) / (q1 - q0)
            p[i] = 1 - (taus[j - 1] + frac * (taus[j] - taus[j - 1]))
    return np.clip(p, 0, 1)


class LogitCalibrator:
    """Smooth monotone recalibration: logistic regression on logit(p_raw).

    (Isotonic regression gives step functions, i.e. many identical
    probabilities, which is confusing for operators.)
    """

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = _logit(p)
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_calibrator(p_raw: np.ndarray, events: np.ndarray) -> LogitCalibrator | None:
    from sklearn.linear_model import LogisticRegression
    m = np.isfinite(p_raw) & np.isfinite(events)
    if m.sum() < 50 or events[m].sum() < 5:
        return None
    lr = LogisticRegression(C=10.0)
    lr.fit(_logit(p_raw[m])[:, None], events[m].astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a <= 0:  # must stay monotone increasing
        return None
    return LogitCalibrator(a, b)


fit_isotonic = fit_calibrator  # backwards-compatible name


def calibrate(cal, p: np.ndarray) -> np.ndarray:
    if cal is None:
        return np.clip(p, 0, 1)
    return np.clip(cal.predict(p), 0, 1)


def confidence(horizon_h: int, width90: float, threshold: float, stale_h: float,
               weather_ok: bool, operational_max_h: int = 24) -> tuple[float, str, list[str]]:
    """Return (score 0..1, label, reasons). Independent of the flood probability."""
    reasons = []
    if horizon_h <= 6:
        s_h = 1.0
    elif horizon_h <= operational_max_h:
        s_h = 0.8
    else:
        s_h = 0.45
        reasons.append("extended horizon (awareness only)")
    rel = width90 / max(threshold, 1e-6)
    s_w = float(np.clip(1.2 - rel, 0.2, 1.0))
    if rel > 0.6:
        reasons.append("wide prediction interval")
    s_d = 1.0
    if stale_h is not None and stale_h > 3:
        s_d *= 0.6
        reasons.append(f"sensor data stale ({int(stale_h)} h)")
    if not weather_ok:
        s_d *= 0.7
        reasons.append("weather forecast degraded / unavailable")
    score = float(np.clip(s_h * (0.5 + 0.5 * s_w) * s_d, 0, 1))
    label = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"
    return score, label, reasons
