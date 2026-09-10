"""
Basic sanity tests for SlopeSense.

Run:  pip install -r requirements-dev.txt
      pytest test_app.py
"""

import os

os.environ.setdefault("SLOPESENSE_API_KEY", "test-key")

from fastapi.testclient import TestClient

from main import app, classify, compute_risk

client = TestClient(app)


def test_classify_boundaries():
    assert classify(0.0) == "Low"
    assert classify(0.30) == "Low"
    assert classify(0.31) == "Moderate"
    assert classify(0.55) == "Moderate"
    assert classify(0.56) == "High"
    assert classify(0.75) == "High"
    assert classify(0.76) == "Critical"
    assert classify(1.0) == "Critical"


def test_compute_risk_is_clamped_to_unit_interval():
    r = compute_risk(slope_factor=1.0, history_factor=1.0, rain72=10_000, intensity=10_000, moisture=10_000)
    assert 0.0 <= r <= 1.0


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["ai_model"] == "logistic_regression_demo"


def test_state_endpoint_returns_all_stations():
    res = client.get("/api/state")
    assert res.status_code == 200
    body = res.json()
    assert len(body["stations"]) == 5
    for s in body["stations"]:
        assert 0.0 <= s["risk"] <= 1.0
        assert s["level"] in ("Low", "Moderate", "High", "Critical")
        assert 0.0 <= s["ai_probability"] <= 1.0


def test_telemetry_rejects_missing_api_key():
    res = client.post("/api/telemetry", json={
        "station_id": "nongpoh", "rain72": 50, "intensity": 3, "moisture": 40,
    })
    assert res.status_code == 401


def test_telemetry_accepts_valid_api_key():
    res = client.post(
        "/api/telemetry",
        json={"station_id": "nongpoh", "rain72": 200, "intensity": 25, "moisture": 90},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["station_id"] == "nongpoh"
    assert body["level"] in ("Low", "Moderate", "High", "Critical")


def test_telemetry_rejects_unknown_station():
    res = client.post(
        "/api/telemetry",
        json={"station_id": "does-not-exist", "rain72": 50, "intensity": 3, "moisture": 40},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 404


def test_reset_requires_api_key():
    res = client.post("/api/reset")
    assert res.status_code == 401


def test_reset_with_valid_key_returns_state():
    res = client.post("/api/reset", headers={"X-API-Key": "test-key"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["stations"]) == 5


def test_telemetry_rejects_invalid_moisture():
    # moisture must be 0-100; Pydantic should reject this before it reaches the DB
    res = client.post(
        "/api/telemetry",
        json={"station_id": "nongpoh", "rain72": 50, "intensity": 3, "moisture": 150},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 422
