# Barcelona Urban Flood Prediction & Transport Monitoring Platform

Functional prototype of a **decision-support tool for TMB operators**: it forecasts water accumulation at monitored
street and metro sections of Barcelona (1 h → 72 h, with uncertainty), estimates which metro lines and bus routes
would be affected, and shows everything on an interactive operational map.

> **Prototype status — read first**
> * **Water-level sensors are SIMULATED** (no public sensor data exists yet). The simulator is physically based and
>   driven by **real** Open-Meteo rainfall. Every synthetic value is flagged `is_synthetic = 1` and the UI shows a
>   *SIMULATED SENSOR DATA* badge. Real sensors plug in through the same database schema.
> * **Vehicle positions are SIMULATED** along **real** OpenStreetMap route geometries (TMB publishes no public live
>   vehicle positions).
> * Forecasts are probabilistic and never guaranteed. Final operational decisions remain with authorised TMB personnel.

---

## 1. Quick start (local, `localhost`)

Requirements: **Python 3.10+** (tested with 3.13), internet access for the first setup.

```bash
git clone https://github.com/llorenccobos/CBI_Project.git
cd CBI_Project
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional API keys (all free, all optional)
python main.py setup          # ~15-25 min: network, weather, sensors, training, evaluation
python main.py serve          # open http://127.0.0.1:8050
```

`setup` uses the OpenStreetMap data and a weather snapshot bundled in the repository when the external APIs are
unavailable, so it also works offline (then run `python main.py update` later to catch up with recent weather).

### Commands

| Command | What it does |
|---|---|
| `python main.py setup [--quick]` | First run: everything below in order (`--quick` skips the learner comparison) |
| `python main.py serve` | Dashboard + API on `http://127.0.0.1:8050`, with hourly automatic updates |
| `python main.py update` | Refresh weather history + live forecast, optional station data, extend sensor series |
| `python main.py network [--offline]` | Rebuild metro / bus layers and monitored segments from OpenStreetMap |
| `python main.py sensors` | Regenerate synthetic sensor series |
| `python main.py train` | Train the probabilistic models |
| `python main.py evaluate [--quick]` | Test-period evaluation → `reports/model_report.md`, `reports/evaluation.json` |
| `python main.py forecast [--issue 2025-09-21T15:00]` | Print a forecast summary (live or replay) |
| `pytest -q` | Unit + API tests |

### API keys (optional, free)

Copy `.env.example` to `.env` and fill what you have. Never commit `.env` (it is git-ignored).

| Variable | Provider | How to get it | Used for |
|---|---|---|---|
| `METEOCAT_API_KEY` | Meteocat XEMA | form at <https://apidocs.meteo.cat> | measured rainfall at Barcelona stations (X4 Raval, X8 Zona Universitària, D5 Fabra) |
| `AEMET_API_KEY` | AEMET OpenData | <https://opendata.aemet.es/centrodedescargas/altaUsuario> | hourly observations (Airport, Barcelona-Fabra) |
| `TMB_APP_ID`, `TMB_APP_KEY` | TMB Developer | register + create an app at <https://developer.tmb.cat> | official GTFS (timetables, stops, shapes) |

Open-Meteo needs no key. The Meteocat / AEMET / TMB clients follow the providers' documentation but could not be
tested without keys during development; their state appears in the dashboard's *Data sources* panel.

---

## 2. Architecture

```
config/            settings.yaml · thresholds.yaml · locations.yaml · settings.py (loader, .env secrets)
data_collection/   meteo_openmeteo.py   gridded history, archived forecasts, live forecast (no key)
                   meteo_stations.py    Meteocat / AEMET station observations (optional keys)
                   transport_osm.py     TMB metro + bus relations, street axes (Overpass, cached)
                   transport_tmb.py     official TMB GTFS (optional keys)
                   sensors_synthetic.py SIMULATED water-level sensors driven by real rainfall
                   weather_service.py   ingestion orchestration + offline seed snapshot
data_processing/   cleaning.py          sensor QC (range, spikes, flat-lines, gaps), hourly sync
                   features.py          feature engineering, forecast-rain reconstruction
data_storage/      db.py                SQLAlchemy Core schema (SQLite now, PostgreSQL/PostGIS-ready)
                   repository.py        read helpers
models/            train.py · predict.py · uncertainty.py · evaluate.py · artifacts/<version>/
geospatial/        network.py           stitching, segments between stations, sensor↔segment↔line links
api/               server.py (FastAPI routes) · services.py (business logic, cache, scheduler job)
dashboard/static/  index.html · app.js · styles.css · assets/logo.svg · vendor/ (MapLibre, Plotly)
main.py            CLI entry point
tests/             pytest suite
```

Data flow: `collect → store (SQLite) → clean/QC → features → LightGBM quantile models → conformal calibration →
probabilities → FastAPI → MapLibre dashboard`. An in-process scheduler (APScheduler) refreshes weather, extends
the sensor series and invalidates cached forecasts every 60 min.

**Why FastAPI + MapLibre instead of Streamlit/Dash:** Streamlit re-runs the whole script on each interaction (slow
slider, flickering map, no smooth vehicle animation). A small REST API plus a static MapLibre front-end gives
instant slider updates (all horizons are delivered in one response), GPU-rendered layers and animation, while all
data and ML logic stays in Python. There is no Node build step; front-end libraries are vendored.

---

## 3. Data sources (Phase 1 findings)

| Source | Content | Resolution | Status / limitations |
|---|---|---|---|
| [Open-Meteo](https://open-meteo.com/en/docs) Historical Forecast API | hourly precipitation, T, RH, pressure, wind, gusts, CAPE | 1.3 km AROME (Météo-France) over Catalonia for the first ~2 days of each run, ECMWF IFS / ICON beyond; hourly | **Used.** Free for non-commercial use (10 000 calls/day per IP); production use by TMB requires the commercial plan |
| [Open-Meteo](https://open-meteo.com/en/docs/previous-runs-api) Previous Runs API | precipitation forecast issued 1, 2, 3 days earlier | as above | **Used** to train with realistic forecast errors. Available from ~2024 (earlier years use a degraded proxy) |
| Open-Meteo Forecast API | live 72 h forecast | as above | **Used** (falls back to the historical-forecast endpoint's latest run when rate-limited) |
| [Meteocat XEMA](https://apidocs.meteo.cat) | measured station data (30-min) | 3 stations in Barcelona | Optional (free key, low monthly quota). Best *observed* rainfall |
| [AEMET OpenData](https://opendata.aemet.es) | station observations, municipal forecasts | 2 stations | Optional (free key). Only last 24 h of hourly observations |
| [OpenStreetMap](https://www.openstreetmap.org) via Overpass | TMB metro route relations (track geometry + stations), 105 TMB bus routes, 5 000+ stops, street axes | very high | **Used** (ODbL). Public Overpass servers are often overloaded; responses are cached in `data/raw/osm` and committed |
| [OpenFreeMap](https://openfreemap.org) | vector basemap (light / dark) | – | **Used** (free, no key; OpenMapTiles schema, © OpenStreetMap contributors) |
| [TMB Developer](https://developer.tmb.cat) | official GTFS, iBus/iMetro arrival predictions | official | Optional (free key). No public real-time vehicle positions |
| Water-level sensors | – | – | **No public data found** (BCASA sewer levels and TMB pumping-well telemetry are not open). Synthetic sensors are used until TMB provides data |
| Drainage network capacity | – | – | Not public → not used (the model does not need it; it can be added as a static feature) |

**Paid data worth considering later:** the Open-Meteo commercial plan (licensing for operational use, higher
quotas) and, for the 0–3 h horizons, radar-based nowcasts (Meteocat / AEMET radar) which would improve short-term
skill more than any model change.

### Monitored locations (initial set)

23 sections defined in `config/locations.yaml`: **11 metro sections** between real consecutive stations (e.g. L1
Hostafrancs–Espanya, L3 Paral·lel–Drassanes, L4 Ciutadella–Bogatell) and **12 street sections** snapped to real OSM
street axes (Gran Via at Pl. Espanya, Av. Paral·lel, Pg. Colom, Rambla de Prim, Zona Franca…). They were chosen
from low-lying areas, coastal sections and old *rieres*; they are a starting proposal to validate with TMB.
Add a location by appending an entry and running `python main.py network && python main.py sensors && python main.py train`.

---

## 4. Machine-learning approach

* **Targets** per sensor and horizon h ∈ {1, 2, 3, 6, 12, 24, 48, 72} h: level at *t+h* and **maximum level in
  (t, t+h]**.
* **Model:** one *global* LightGBM model per horizon and target (all sensors pooled, sensor characteristics as
  features), **quantile regression** at 8 quantiles (5 %…98 %).
* **Features:** current/lagged levels, trends, observed rainfall (1–72 h), antecedent wetness index, **forecast
  rainfall inside the window**, upstream (Collserola-side) rainfall, levels at the 3 nearest sensors (spatial
  relationships), meteorology (T, RH, pressure tendency, wind, CAPE), time of day / season, location statistics.
* **No look-ahead at training time:** future rainfall inputs are reconstructed from **archived forecast runs**
  (previous-runs API), blended with observations only for the first hours to represent nowcast skill.
* **Uncertainty:** quantiles → re-sorted → **conformalized quantile regression** (separate dry/wet regimes) for a
  calibrated 90 % interval → P(level > threshold) read from the predictive distribution (works for any threshold,
  so thresholds can be edited without retraining) → logistic recalibration. **Confidence** (horizon tier, interval
  width vs threshold, data freshness) is reported separately and is *not* the flood probability.
* **Alarm rule:** an alert is raised when P(max in window > alert threshold) exceeds a cut-off tuned on the
  calibration period to keep **recall ≥ 90 %** (missed floods are costlier than false alarms); the false-alarm
  ratio is reported.
* **Splits:** chronological train / validation / calibration / test with 72 h gaps. If the data does not cover the
  configured dates, proportional chronological splits are used (shown in the report).
* **Evaluation** (`reports/model_report.md`): MAE/RMSE (all and wet hours), window-maximum error, interval coverage,
  precision / recall / FNR / false-alarm ratio, Brier score + reliability curve, event-level detection by lead
  time; baselines: persistence, climatology, rain-driven linear model; learner comparison: LightGBM vs XGBoost vs
  CatBoost vs Random Forest, plus an ablation without spatial features.

---

## 5. Dashboard guide

* **Map:** real metro lines (official colours), TMB bus routes, stations / stops, monitored segments coloured
  green / yellow / orange / red. The **glow width** shows forecast uncertainty and **dashed** segments mark the
  48–72 h *awareness* horizons.
* **Time slider:** `Now → +1h … +72h` (▶ animates). *Map value* switches between the expected level at the time,
  the **maximum within the window**, and the worst case (95 % upper bound).
* **Filters:** metro / street sections, risk categories, transport layers, simulated vehicles.
* **Weather layer:** rainfall intensity, accumulated rainfall, temperature, wind, humidity (independent ≈3 km grid).
* **Forecast mode:** *Live* or *Replay past event*: re-issues the forecast as it would have been made before a past
  storm and overlays what was measured afterwards (events are labelled in-sample / out-of-sample).
* **Detail panel** (click a segment): KPIs, measured vs forecast chart with 90 % band and rainfall, expected time of
  threshold exceedance, factor attributions ("why"), table by horizon, affected lines and connected stations / stops.

### Configuration

* **Thresholds:** `config/thresholds.yaml` (defaults 10 / 20 / **30 cm**, per-type and per-sensor overrides; the
  alert level is `critical`). Changes apply immediately, no retraining needed.
* **Logo:** replace `dashboard/static/assets/logo.svg` (or point `branding.logo_path` in `config/settings.yaml` to
  a PNG/SVG under `dashboard/static/`). The TMB logo is intentionally not included.
* **Port, update interval, horizons, grid, model parameters:** `config/settings.yaml`.

---

## 6. Moving to real data

1. Write the readings into `sensor_observations` (`sensor_id, ts (UTC, hourly), level_cm, quality, is_synthetic=0`)
   — e.g. an adapter in `data_collection/` that reads TMB's SCADA export.
2. Update `config/locations.yaml` with the real sensor ids / positions, run `python main.py network`.
3. Stop running `sensors` (or remove the synthetic rows), then `python main.py train && python main.py evaluate`.
4. Migrate storage to PostgreSQL/PostGIS by changing `storage.url` (SQLAlchemy Core is used throughout).

## 7. Current results (synthetic sensors, test period 2026-01 → 2026-11, out-of-sample)

Full report: `reports/model_report.md` (also in the dashboard under *Model performance*).

* Median-level error beats persistence from +2 h onwards (wet-hours MAE at +6 h: 1.05 cm vs 1.59 cm; at +24 h: 1.29 vs 2.87).
* 90 % intervals cover 89–90 % of all hours (85–88 % in wet hours: slightly over-confident during storms).
* Exceedance of 30 cm (window maximum): recall 0.88–0.96 with precision 0.72 (+1 h) down to 0.33 (+72 h);
  57–62 of 65 flood onsets detected depending on lead time.
* **Weaknesses found:** the median under-estimates flood *peaks* (window-max error ≈ 8.5 cm on eventful windows;
  persistence is better at +1–3 h), because the quantile median is pulled towards the many dry hours. Probabilities
  and alarms compensate, but peak magnitude should be read from the upper bound. Random Forest had lower wet-hour MAE
  at 1–6 h in the learner comparison; LightGBM is kept for native quantile regression, speed and attributions.
  Spatial features gave no measurable gain on synthetic data (expected: the simulator has no inter-sensor coupling).

## 8. Known limitations

* Synthetic sensors: model metrics demonstrate the pipeline, not real-world skill.
* Weather grid ≈ 3 km and hourly; convective cells are smaller and faster → radar nowcasting is the main upgrade path.
* Forecast-error reconstruction before 2024 uses a degraded proxy (no archived runs available).
* The free Open-Meteo tier is rate-limited per IP; in shared networks the live forecast may be temporarily
  unavailable (the dashboard flags it and lowers confidence).
