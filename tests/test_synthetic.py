import numpy as np

from data_collection.sensors_synthetic import sensor_params, simulate_true_level


def test_generator_deterministic_and_prefix_stable():
    rng = np.random.default_rng(1)
    P = rng.gamma(0.05, 8, size=2000)
    p = sensor_params("TEST-1", "street", "high")
    assert p == sensor_params("TEST-1", "street", "high")
    a = simulate_true_level(P, P, "street", p)
    b = simulate_true_level(P[:1500], P[:1500], "street", p)
    assert np.allclose(a[:1500], b)


def test_dry_weather_gives_no_water():
    P = np.zeros(500)
    for kind in ("street", "metro"):
        p = sensor_params("TEST-2", kind, "medium")
        assert simulate_true_level(P, P, kind, p).max() < 1e-9


def test_storm_raises_level():
    P = np.zeros(200); P[50:53] = [20, 45, 15]
    p = sensor_params("TEST-3", "street", "high")
    s = simulate_true_level(P, P, "street", p)
    assert s[52] > s[49] and s[120] < s[53]
