"""API smoke tests. Require the database and a trained model (python main.py setup)."""
import pytest

from config.settings import ARTIFACTS_DIR, DATA_DIR

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "latest.txt").exists() or not (DATA_DIR / "flood.db").exists(),
                                reason="requires setup (database + trained model)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


def test_meta_and_layers(client):
    m = client.get("/api/meta").json()
    assert m["horizons"] == [1, 2, 3, 6, 12, 24, 48, 72]
    for name in ("metro_lines", "bus_routes", "segments", "weather_grid"):
        r = client.get(f"/api/layers/{name}")
        assert r.status_code == 200 and r.json()["features"]
    assert client.get("/api/layers/nope").status_code == 404


def test_live_forecast_structure(client):
    fc = client.get("/api/forecast").json()
    assert fc["synthetic_sensors"] is True and fc["sensors"]
    for s in fc["sensors"]:
        assert len(s["horizons"]) == 8
        for h in s["horizons"]:
            lv = h["level"]
            assert lv["lo"] <= lv["median"] <= lv["hi"]
            p = h["p_exceed_max"]
            assert 0 <= p["critical"] <= p["dangerous"] + 1e-6 <= p["elevated"] + 2e-6 <= 1 + 2e-6
            assert h["confidence"]["label"] in ("high", "medium", "low")


def test_replay_and_detail(client):
    ev = client.get("/api/events").json()
    assert isinstance(ev, list)
    issue = ev[0]["issue_suggested"] if ev else None
    fc = client.get("/api/forecast", params={"issue": issue} if issue else {}).json()
    sid = fc["sensors"][0]["sensor_id"]
    d = client.get(f"/api/sensor/{sid}", params={"issue": issue} if issue else {}).json()
    assert d["history"] and d["forecast"]["sensor_id"] == sid
    if issue:
        assert fc["mode"] == "replay" and d["observed_after"]


def test_weather_layer(client):
    wx = client.get("/api/weather").json()
    assert "0" in wx["horizons"] and len(wx["horizons"]["0"]) > 0
