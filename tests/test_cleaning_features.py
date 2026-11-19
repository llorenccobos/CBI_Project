import numpy as np
import pandas as pd

from data_processing.cleaning import hours_since_valid, qc_flags
from data_processing.features import LMAX, _lead_weights, window_features


def test_qc_detects_spike_flatline_missing():
    idx = pd.date_range("2025-01-01", periods=30, freq="h")
    x = pd.Series(0.3, index=idx)
    x.iloc[5] = 20          # spike without rain
    x.iloc[10:16] = 7.5     # frozen reading
    x.iloc[20] = np.nan
    q = qc_flags(x, pd.Series(0.0, index=idx))
    assert q.iloc[5] == "suspect"
    assert (q.iloc[10:16] == "suspect").all()
    assert q.iloc[20] == "missing"
    assert q.iloc[0] == "good"


def test_spike_with_rain_is_kept():
    idx = pd.date_range("2025-01-01", periods=10, freq="h")
    x = pd.Series(0.3, index=idx); x.iloc[5] = 20
    rain = pd.Series(0.0, index=idx); rain.iloc[5] = 12
    assert qc_flags(x, rain).iloc[5] == "good"


def test_hours_since_valid():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 2.0]})
    assert hours_since_valid(df)["a"].tolist() == [0, 1, 2, 0]


def test_lead_weights_sum_to_one():
    for L in range(1, LMAX + 1):
        assert abs(sum(_lead_weights(L)) - 1) < 1e-9


def test_window_features():
    F = np.zeros((2, LMAX), dtype=np.float32)
    F[0, :3] = [1, 2, 3]
    w = window_features(F, 3)
    assert w["fc_sum"][0] == 6 and w["fc_max"][0] == 3 and w["fc_next1"][0] == 1
    assert w["fc_sum"][1] == 0
