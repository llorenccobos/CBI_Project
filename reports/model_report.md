# Model evaluation report

Model version **20261119-115949** · test period 2026-01-04 00:00:00 → 2026-11-19 10:00:00

> Sensor data are **SYNTHETIC** (physically-based simulator driven by real Open-Meteo rainfall).
> Metrics show that the pipeline works end-to-end; they are not evidence of real-world skill.

## Water level (median forecast) — MAE / RMSE in cm

| Horizon | LightGBM MAE | wet MAE | Persistence wet MAE | Climatology wet MAE | Rain-linear wet MAE | 90% coverage | 90% coverage (wet) |
|---|---|---|---|---|---|---|---|
| +1 h | 0.34 | 0.76 | 0.65 | 2.17 | 0.61 | 0.90 | 0.87 |
| +2 h | 0.37 | 0.81 | 0.89 | 2.15 | 0.82 | 0.90 | 0.87 |
| +3 h | 0.40 | 0.89 | 1.09 | 2.14 | 1.01 | 0.90 | 0.88 |
| +6 h | 0.47 | 1.05 | 1.59 | 2.10 | 1.41 | 0.90 | 0.87 |
| +12 h | 0.54 | 1.19 | 2.25 | 2.06 | 1.72 | 0.89 | 0.87 |
| +24 h | 0.59 | 1.29 | 2.87 | 2.03 | 1.85 | 0.89 | 0.86 |
| +48 h | 0.67 | 1.46 | 3.11 | 2.02 | 1.95 | 0.89 | 0.86 |
| +72 h | 0.71 | 1.56 | 3.24 | 2.01 | 1.98 | 0.89 | 0.85 |

## Threshold exceedance (window maximum ≥ alert threshold)

| Horizon | Alarm p* | Events | Recall | FNR | Precision | False-alarm ratio | Brier | Persistence recall |
|---|---|---|---|---|---|---|---|---|
| +1 h | 0.09 | 765 | 0.89 | 0.11 | 0.72 | 0.28 | 0.0018 | 0.91 |
| +2 h | 0.14 | 832 | 0.88 | 0.12 | 0.72 | 0.28 | 0.0019 | 0.84 |
| +3 h | 0.08 | 899 | 0.94 | 0.06 | 0.66 | 0.34 | 0.0023 | 0.78 |
| +6 h | 0.02 | 1088 | 0.92 | 0.08 | 0.55 | 0.45 | 0.0033 | 0.64 |
| +12 h | 0.03 | 1454 | 0.96 | 0.04 | 0.52 | 0.48 | 0.0042 | 0.48 |
| +24 h | 0.02 | 2195 | 0.95 | 0.05 | 0.41 | 0.59 | 0.0061 | 0.32 |
| +48 h | 0.02 | 3671 | 0.96 | 0.04 | 0.32 | 0.68 | 0.0119 | 0.19 |
| +72 h | 0.02 | 5182 | 0.96 | 0.04 | 0.33 | 0.67 | 0.0193 | 0.14 |

## Window-maximum error on eventful windows (max ≥ 10 cm)

| Horizon | LightGBM MAE | Persistence MAE |
|---|---|---|
| +1 h | 8.52 | 2.99 |
| +2 h | 8.46 | 4.42 |
| +3 h | 8.39 | 5.67 |
| +6 h | 8.35 | 8.65 |
| +12 h | 8.48 | 12.46 |
| +24 h | 8.21 | 16.35 |
| +48 h | 9.59 | 19.48 |
| +72 h | 10.61 | 21.19 |

## Event-level detection (65 flood onsets in test period)

| Lead time | Detected / total |
|---|---|
| 1 h ahead | 62 / 65 |
| 2 h ahead | 48 / 65 |
| 3 h ahead | 50 / 65 |
| 6 h ahead | 54 / 65 |
| 12 h ahead | 60 / 65 |
| 24 h ahead | 55 / 65 |
| 48 h ahead | 53 / 65 |
| 72 h ahead | 59 / 65 |

## Learner comparison (median level, wet-hours MAE in cm)

| Horizon | LightGBM | XGBoost | CatBoost | RandomForest | LightGBM (no spatial features) |
|---|---|---|---|---|---|
| +1 h | 0.65 | 0.69 | 0.70 | 0.51 | 0.65 |
| +3 h | 0.82 | 0.83 | 0.83 | 0.69 | 0.81 |
| +6 h | 1.01 | 0.98 | 1.01 | 0.91 | 1.00 |
| +24 h | 1.26 | 1.27 | 1.30 | 1.29 | 1.26 |

## Most important features (gain, median model)

* **h1_y_lvl**: lvl_max6, lvl_0, lvl_l1, is_metro, s_mean, lvl_mean24, fc_sum, lvl_d1
* **h6_y_max**: lvl_max6, lvl_0, lvl_mean24, lvl_l1, is_metro, s_mean, fc_sum, fc_max
* **h24_y_max**: lvl_max6, fc_sum, lvl_mean24, lvl_0, fcu_sum, s_mean, fc_max, is_metro