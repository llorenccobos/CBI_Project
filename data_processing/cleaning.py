"""Quality control of water-level series and synchronisation to an hourly grid.

Rules (applied per sensor, on raw measurements):
  * range check      : values outside [0, 300] cm           -> suspect
  * spike detection  : jump > 5 cm that reverts next hour, with
                       no rain to explain it                -> suspect
  * flat-line check  : identical non-zero value >= 4 h      -> suspect
  * missing values   : NaN                                  -> missing

Suspect / missing values are excluded from model inputs and targets; short
gaps (<= 3 h) in model inputs are forward-filled, longer gaps stay NaN and are
signalled to operators as "stale sensor".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qc_flags(levels: pd.Series, rain: pd.Series | None = None) -> pd.Series:
    """Return quality flags for one sensor's hourly series (index = ts)."""
    x = levels.astype(float)
    q = pd.Series("good", index=x.index, dtype=object)
    q[x.isna()] = "missing"
    q[(x < 0) | (x > 300)] = "suspect"
    prev, nxt = x.shift(1), x.shift(-1)
    rain_ok = (rain.reindex(x.index).fillna(0) > 0.5) if rain is not None else pd.Series(False, index=x.index)
    spike = ((x - prev) > 5) & ((x - nxt) > 5) & ~rain_ok
    q[spike] = "suspect"
    grp = (x != x.shift()).cumsum()
    run_len = x.groupby(grp).transform("size")
    flat = (run_len >= 4) & (x > 1)
    q[flat & (q == "good")] = "suspect"
    return q


def clean_wide(obs: pd.DataFrame, rain_by_sensor: dict[str, pd.Series] | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """obs (long: sensor_id, ts, level_cm) -> (clean wide levels, wide quality flags).

    Clean levels contain NaN where data is missing or suspect.
    """
    wide = obs.pivot_table(index="ts", columns="sensor_id", values="level_cm", aggfunc="mean")
    wide = wide.sort_index().asfreq("h")
    flags = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=object)
    for sid in wide.columns:
        rain = rain_by_sensor.get(sid) if rain_by_sensor else None
        flags[sid] = qc_flags(wide[sid], rain)
    clean = wide.where(flags == "good")
    return clean, flags


def ffill_limited(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    return df.ffill(limit=limit)


def hours_since_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Hours since the last valid reading (0 if current reading valid)."""
    out = {}
    for c in df.columns:
        valid = df[c].notna().to_numpy()
        idx = np.where(valid, np.arange(len(valid)), -1)
        last = np.maximum.accumulate(idx)
        h = np.where(last >= 0, np.arange(len(valid)) - last, 9999)
        out[c] = h
    return pd.DataFrame(out, index=df.index)
