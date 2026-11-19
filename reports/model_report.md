# Model evaluation report

Model version **20261119-111901** · test period 2025-05-17 01:00:00 → 2026-01-03 23:00:00

> Sensor data are **SYNTHETIC** (physically-based simulator driven by real Open-Meteo rainfall).
> Metrics show that the pipeline works end-to-end; they are not evidence of real-world skill.

## Water level (median forecast) — MAE / RMSE in cm

| Horizon | LightGBM MAE | wet MAE | Persistence wet MAE | Climatology wet MAE | Rain-linear wet MAE | 90% coverage | 90% coverage (wet) |
|---|---|---|---|---|---|---|---|
| +1 h | 0.27 | 0.57 | 0.62 | 1.56 | 0.58 | 0.89 | 0.86 |
| +2 h | 0.29 | 0.63 | 0.83 | 1.55 | 0.73 | 0.89 | 0.86 |
| +3 h | 0.31 | 0.66 | 1.01 | 1.54 | 0.87 | 0.90 | 0.86 |
| +6 h | 0.37 | 0.81 | 1.43 | 1.51 | 1.19 | 0.89 | 0.86 |
| +12 h | 0.46 | 1.01 | 1.93 | 1.49 | 1.47 | 0.89 | 0.84 |
| +24 h | 0.51 | 1.11 | 2.22 | 1.47 | 1.59 | 0.88 | 0.83 |
| +48 h | 0.55 | 1.18 | 2.35 | 1.47 | 1.69 | 0.89 | 0.84 |
| +72 h | 0.58 | 1.24 | 2.41 | 1.46 | 1.71 | 0.88 | 0.82 |

## Threshold exceedance (window maximum ≥ alert threshold)

| Horizon | Alarm p* | Events | Recall | FNR | Precision | False-alarm ratio | Brier | Persistence recall |
|---|---|---|---|---|---|---|---|---|
| +1 h | 0.08 | 329 | 0.85 | 0.15 | 0.84 | 0.16 | 0.0012 | 0.91 |
| +2 h | 0.09 | 360 | 0.89 | 0.11 | 0.77 | 0.23 | 0.0012 | 0.83 |
| +3 h | 0.06 | 391 | 0.92 | 0.08 | 0.72 | 0.28 | 0.0013 | 0.76 |
| +6 h | 0.02 | 479 | 0.96 | 0.04 | 0.64 | 0.36 | 0.0016 | 0.62 |
| +12 h | 0.02 | 651 | 0.94 | 0.06 | 0.54 | 0.46 | 0.0026 | 0.45 |
| +24 h | 0.02 | 1013 | 0.83 | 0.17 | 0.47 | 0.53 | 0.0047 | 0.30 |
| +48 h | 0.02 | 1709 | 0.87 | 0.13 | 0.46 | 0.54 | 0.0073 | 0.18 |
| +72 h | 0.02 | 2405 | 0.91 | 0.09 | 0.29 | 0.71 | 0.0116 | 0.13 |

## Window-maximum error on eventful windows (max ≥ 10 cm)

| Horizon | LightGBM MAE | Persistence MAE |
|---|---|---|
| +1 h | 7.77 | 3.35 |
| +2 h | 7.94 | 4.99 |
| +3 h | 8.05 | 6.44 |
| +6 h | 8.45 | 9.74 |
| +12 h | 9.60 | 13.56 |
| +24 h | 10.91 | 17.01 |
| +48 h | 13.31 | 19.92 |
| +72 h | 12.82 | 21.32 |

## Event-level detection (31 flood onsets in test period)

| Lead time | Detected / total |
|---|---|
| 1 h ahead | 25 / 31 |
| 2 h ahead | 23 / 31 |
| 3 h ahead | 15 / 31 |
| 6 h ahead | 26 / 31 |
| 12 h ahead | 26 / 31 |
| 24 h ahead | 22 / 31 |
| 48 h ahead | 25 / 31 |
| 72 h ahead | 2 / 31 |

## Learner comparison (median level, wet-hours MAE in cm)

| Horizon | LightGBM | XGBoost | CatBoost | RandomForest | LightGBM (no spatial features) |
|---|---|---|---|---|---|
| +1 h | 0.50 | 0.53 | 0.51 | 0.42 | 0.49 |
| +3 h | 0.62 | 0.62 | 0.64 | 0.54 | 0.61 |
| +6 h | 0.79 | 0.77 | 0.79 | 0.73 | 0.79 |
| +24 h | 1.10 | 1.07 | 1.12 | 1.06 | 1.09 |

## Most important features (gain, median model)

* **h1_y_lvl**: lvl_max6, lvl_0, lvl_l1, is_metro, s_mean, lvl_mean24, fc_sum, lvl_mean168
* **h6_y_max**: lvl_max6, lvl_0, lvl_l1, is_metro, lvl_mean24, s_mean, fc_sum, lvl_mean168
* **h24_y_max**: lvl_max6, fc_sum, lvl_mean24, lvl_0, fcu_sum, fc_max, is_metro, s_mean