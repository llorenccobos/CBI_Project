"""Feature engineering for multi-horizon water-level forecasting.

One row = (sensor, issue time t). For a horizon h the model predicts the
level at t+h and the maximum level over (t, t+h].

Feature groups (also used to explain predictions to operators):
  level        current level, lags, recent max / mean, staleness
  trend        1 h / 3 h level changes
  rain_obs     observed rainfall over the last 1..72 h, peak intensity
  wetness      antecedent precipitation index (soil / network saturation)
  rain_fc      FORECAST rainfall inside the prediction window
  upstream     observed + forecast rainfall in the uphill (Collserola) cell
  neighbours   levels at the 3 nearest other sensors (spatial information)
  meteo        temperature, humidity, pressure tendency, wind, CAPE
  time         hour of day, season
  location     infrastructure type, elevation, sensor statistics

Forecast rainfall at training time
----------------------------------
To avoid training on "perfect" future rainfall, the forecast available at
issue time t for valid time t+L is reconstructed from Open-Meteo's archive of
previous model runs (forecasts issued 1, 2 and 3 days before):

  L <= 24 h : w*observed + (1-w)*day-1 forecast, w = exp(-L/6)   (nowcast skill decays in hours)
  24..48 h  : blend day-1 -> day-2 forecast
  48..72 h  : blend day-2 -> day-3 forecast

Where archived runs are not available (before 2024), a degraded proxy
(temporally smoothed observed rain with multiplicative noise) is used.
In live operation the real forecast run is used instead.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data_processing.cleaning import clean_wide, ffill_limited, hours_since_valid
from data_storage.repository import load_observations, load_sensors, load_weather

LMAX = 72
EPS = 1e-6

FEATURE_GROUPS = {
    "level": ["lvl_0", "lvl_l1", "lvl_l2", "lvl_l3", "lvl_l6", "lvl_l12", "lvl_l24", "lvl_max6", "lvl_max24",
              "lvl_mean24", "lvl_mean168", "stale_h"],
    "trend": ["lvl_d1", "lvl_d3"],
    "rain_obs": ["rain_1h", "rain_3h", "rain_6h", "rain_12h", "rain_24h", "rain_72h", "rain_max6"],
    "wetness": ["api"],
    "rain_fc": ["fc_sum", "fc_max", "fc_last3", "fc_last6", "fc_next1", "fc_next3", "fc_next6"],
    "upstream": ["up_rain_3h", "up_rain_6h", "fcu_sum"],
    "neighbours": ["nb_mean", "nb_max", "nb_d1"],
    "meteo": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "dp3", "wind_speed_10m",
              "wind_gusts_10m", "cape"],
    "time": ["hod_sin", "hod_cos", "doy_sin", "doy_cos", "thod_sin", "thod_cos"],
    "location": ["is_metro", "elevation_m", "s_p99", "s_mean"],
}
GROUP_LABELS = {
    "level": "Current water level", "trend": "Recent level trend", "rain_obs": "Observed rainfall (recent hours)",
    "wetness": "Antecedent wetness", "rain_fc": "Forecast rainfall in window", "upstream": "Upstream rainfall",
    "neighbours": "Neighbouring sensors", "meteo": "Meteorological conditions", "time": "Time of day / season",
    "location": "Location / sensor behaviour",
}
SPATIAL_FEATURES = FEATURE_GROUPS["neighbours"] + FEATURE_GROUPS["upstream"]
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
MET_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_gusts_10m", "cape"]


# --------------------------------------------------------------------------- forecast proxy
def _lead_weights(L: int):
    """Return (w_obs, w_d1, w_d2, w_d3) for lead time L (hours)."""
    if L <= 24:
        w = float(np.exp(-L / 6.0))
        return w, 1 - w, 0.0, 0.0
    if L <= 48:
        a = (L - 24) / 24
        return 0.0, 1 - a, a, 0.0
    a = (L - 48) / 24
    return 0.0, 0.0, 1 - a, a


def _degraded(obs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fallback 'k days ahead' forecast: smoothed observed rain with multiplicative noise."""
    s = pd.Series(obs).rolling(6 * k + 1, center=True, min_periods=1).mean().to_numpy()
    rng = np.random.default_rng(seed + k)
    return s * np.exp(rng.normal(0, 0.45, size=len(s)) - 0.1)


def window_features(F: np.ndarray, h: int) -> dict:
    """F: (n, LMAX) forecast precipitation, column L-1 = valid time t+L."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        win = F[:, :h]
        out = {
            "fc_sum": np.nansum(win, axis=1),
            "fc_max": np.nanmax(win, axis=1) if h > 0 else np.zeros(len(F)),
            "fc_last3": np.nansum(F[:, max(0, h - 3):h], axis=1),
            "fc_last6": np.nansum(F[:, max(0, h - 6):h], axis=1),
            "fc_next1": np.nansum(F[:, :1], axis=1),
            "fc_next3": np.nansum(F[:, :3], axis=1),
            "fc_next6": np.nansum(F[:, :6], axis=1),
        }
        allnan = np.all(np.isnan(win), axis=1)
        for k in ("fc_sum", "fc_max", "fc_last3", "fc_last6"):
            out[k] = np.where(allnan, np.nan, out[k])
    return out


# --------------------------------------------------------------------------- store
class FeatureStore:
    """Aligned hourly arrays (sensors x time, cells x time) for feature building."""

    def __init__(self, start=None, end=None, sensor_stats: dict | None = None):
        self.sensors = load_sensors().set_index("sensor_id")
        obs = load_observations(start=start, end=end)
        if obs.empty:
            raise RuntimeError("No sensor observations available.")
        w = load_weather(start=start, end=(pd.Timestamp(end) + pd.Timedelta(hours=LMAX)) if end is not None else None)
        self.index = pd.date_range(obs.ts.min(), obs.ts.max(), freq="h")
        self.ext_index = pd.date_range(self.index[0], self.index[-1] + pd.Timedelta(hours=LMAX), freq="h")
        self.cells = sorted(w.cell_id.unique())
        self.weather = {}
        for var in ["precipitation", "precip_d1", "precip_d2", "precip_d3", *MET_VARS]:
            self.weather[var] = (w.pivot_table(index="ts", columns="cell_id", values=var, aggfunc="mean")
                                 .reindex(index=self.ext_index, columns=self.cells))
        rain_by_sensor = {sid: self.weather["precipitation"][s.weather_cell].reindex(self.index)
                          for sid, s in self.sensors.iterrows()}
        clean, flags = clean_wide(obs, rain_by_sensor)
        clean = clean.reindex(index=self.index, columns=self.sensors.index)
        self.flags = flags.reindex(index=self.index, columns=self.sensors.index)
        self.clean = clean                      # NaN where missing / suspect
        self.filled = ffill_limited(clean, 3)   # inputs
        self.stale = hours_since_valid(clean)
        self.sensor_stats = sensor_stats or {}
        self._neighbours = self._compute_neighbours()
        self._fc_cache: dict = {}

    # ----------------------------------------------------------------- helpers
    def _compute_neighbours(self, k: int = 3) -> dict:
        s = self.sensors
        xy = np.c_[s.lon.to_numpy() * 0.75, s.lat.to_numpy()]
        out = {}
        for i, sid in enumerate(s.index):
            d = np.hypot(*(xy - xy[i]).T)
            order = [j for j in np.argsort(d) if j != i][:k]
            out[sid] = list(s.index[order])
        return out

    def compute_sensor_stats(self, until) -> dict:
        part = self.clean.loc[:until]
        return {sid: {"s_p99": float(np.nanquantile(part[sid], 0.99)) if part[sid].notna().any() else np.nan,
                      "s_mean": float(np.nanmean(part[sid])) if part[sid].notna().any() else np.nan}
                for sid in part.columns}

    def proxy_forecast_matrix(self, cell: str) -> np.ndarray:
        """Forecast precipitation available at each issue time (positions of self.index) for leads 1..72."""
        if cell in self._fc_cache:
            return self._fc_cache[cell]
        obs = self.weather["precipitation"][cell].to_numpy(dtype=float)
        seed = sum(map(ord, cell)) * 7919  # deterministic per cell
        d = []
        for k in (1, 2, 3):
            arr = self.weather[f"precip_d{k}"][cell].to_numpy(dtype=float)
            arr = np.where(np.isnan(arr), _degraded(obs, k, seed), arr)
            d.append(arr)
        n = len(self.index)
        F = np.full((n, LMAX), np.nan, dtype=np.float32)
        pos = np.arange(n)
        for L in range(1, LMAX + 1):
            v = pos + L
            wo, w1, w2, w3 = _lead_weights(L)
            val = wo * np.nan_to_num(obs[v]) + w1 * d[0][v] + w2 * d[1][v] + w3 * d[2][v]
            val = np.where(np.isnan(obs[v]) & (wo > 0.5), np.nan, val)
            F[:, L - 1] = val
        self._fc_cache[cell] = F
        return F

    # ----------------------------------------------------------------- features
    def base_features(self, sid: str) -> pd.DataFrame:
        s = self.sensors.loc[sid]
        L = self.filled[sid]
        idx = self.index
        P = self.weather["precipitation"][s.weather_cell].reindex(idx).fillna(0.0)
        Pu = self.weather["precipitation"][s.upstream_cell].reindex(idx).fillna(0.0)
        f = pd.DataFrame(index=idx)
        f["lvl_0"] = L
        for k in (1, 2, 3, 6, 12, 24):
            f[f"lvl_l{k}"] = L.shift(k)
        f["lvl_d1"] = L - L.shift(1)
        f["lvl_d3"] = L - L.shift(3)
        f["lvl_max6"] = L.rolling(6, min_periods=1).max()
        f["lvl_max24"] = L.rolling(24, min_periods=1).max()
        f["lvl_mean24"] = L.rolling(24, min_periods=6).mean()
        f["lvl_mean168"] = L.rolling(168, min_periods=24).mean()
        f["stale_h"] = self.stale[sid].clip(upper=999)
        nb = self.filled[self._neighbours[sid]]
        f["nb_mean"] = nb.mean(axis=1)
        f["nb_max"] = nb.max(axis=1)
        f["nb_d1"] = (nb - nb.shift(1)).mean(axis=1)
        for k in (1, 3, 6, 12, 24, 72):
            f[f"rain_{k}h"] = P.rolling(k, min_periods=1).sum()
        f["rain_max6"] = P.rolling(6, min_periods=1).max()
        f["api"] = P.ewm(alpha=0.03, adjust=False).mean() / 0.03
        f["up_rain_3h"] = Pu.rolling(3, min_periods=1).sum()
        f["up_rain_6h"] = Pu.rolling(6, min_periods=1).sum()
        for var in MET_VARS:
            f[var] = self.weather[var][s.weather_cell].reindex(idx)
        f["dp3"] = f["surface_pressure"] - f["surface_pressure"].shift(3)
        hod = idx.hour.to_numpy()
        doy = idx.dayofyear.to_numpy()
        f["hod_sin"], f["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
        f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        f["is_metro"] = 1.0 if s.infra_type == "metro" else 0.0
        f["elevation_m"] = s.elevation_m if pd.notna(s.elevation_m) else np.nan
        st = self.sensor_stats.get(sid, {})
        f["s_p99"] = st.get("s_p99", np.nan)
        f["s_mean"] = st.get("s_mean", np.nan)
        return f

    def targets(self, sid: str, h: int) -> tuple[pd.Series, pd.Series]:
        c = self.clean[sid]
        y_lvl = c.shift(-h)
        y_max = c[::-1].rolling(h, min_periods=max(1, h // 2)).max()[::-1].shift(-1)
        return y_lvl, y_max

    def horizon_frame(self, h: int, sensors: list[str] | None = None,
                      forecast_override: dict[str, np.ndarray] | None = None,
                      positions: np.ndarray | None = None, base_cache: dict | None = None) -> pd.DataFrame:
        """Features (+ targets) for horizon h.

        forecast_override: {cell_id: (n_positions, LMAX) matrix} to use a real forecast run.
        positions: integer positions in self.index to keep (default: all).
        """
        sensors = sensors or list(self.sensors.index)
        frames = []
        for sid in sensors:
            s = self.sensors.loc[sid]
            base = base_cache[sid] if base_cache is not None and sid in base_cache else self.base_features(sid)
            pos = np.arange(len(self.index)) if positions is None else positions
            f = base.iloc[pos].copy()
            if forecast_override is not None:
                Fl, Fu = forecast_override[s.weather_cell], forecast_override[s.upstream_cell]
            else:
                Fl = self.proxy_forecast_matrix(s.weather_cell)[pos]
                Fu = self.proxy_forecast_matrix(s.upstream_cell)[pos]
            for k, v in window_features(Fl, h).items():
                f[k] = v
            f["fcu_sum"] = window_features(Fu, h)["fc_sum"]
            thod = ((f.index.hour.to_numpy() + h) % 24)
            f["thod_sin"], f["thod_cos"] = np.sin(2 * np.pi * thod / 24), np.cos(2 * np.pi * thod / 24)
            y_lvl, y_max = self.targets(sid, h)
            f["y_lvl"] = y_lvl.iloc[pos].to_numpy()
            f["y_max"] = y_max.iloc[pos].to_numpy()
            f["sensor_id"] = sid
            f["ts"] = f.index
            frames.append(f.reset_index(drop=True))
        out = pd.concat(frames, ignore_index=True)
        num = [c for c in out.columns if c not in ("sensor_id", "ts")]
        out[num] = out[num].astype("float32")
        return out
