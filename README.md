# SlopeSense — AI Landslide Early Warning Prototype

SlopeSense is a working local prototype for monitoring landslide-prone road sections in the Northeast Region. It combines a **FastAPI backend**, SQLite persistence, simulated sensor readings, a lightweight machine-learning prediction layer, an explainable risk view, a regional map, and automated alerts.

## New features in this version

- **FastAPI backend:** typed request/response models (Pydantic), automatic validation, and interactive API docs at `/docs`.
- **Real sensor ingestion:** `POST /api/telemetry` accepts real rainfall/intensity/moisture readings from an actual sensor or weather API, not just the built-in simulator (see below).
- **API-key protection:** the ingestion endpoint requires an `X-API-Key` header so not just anyone can inject fake readings.
- **Alert delivery hook:** when risk escalates to High/Critical, `notifications.py` fires — it always logs the alert, and can optionally forward it to a real webhook (Slack, Discord, Zapier/n8n, etc.) if `ALERT_WEBHOOK_URL` is set.
- **Model evaluation script:** `evaluate_model.py` trains on synthetic data and reports accuracy, precision, recall, F1, and ROC-AUC on a held-out set — no external ML libraries required.
- **Basic test suite:** `test_app.py` covers the risk classification boundaries, the API endpoints, and the new auth/validation logic.
- **AI prediction:** dependency-free logistic regression trained at startup on synthetic labelled examples.
- **Explainable AI:** the dashboard shows the strongest factors influencing the prediction.
- **Prediction window:** estimates landslide probability for the next 6 hours using current and forecast-style inputs.
- **Regional map:** monitoring stations are plotted using Leaflet/OpenStreetMap.
- **Station metadata:** slope angle, elevation, drainage condition and soil type are included.
- **Recommended action:** each risk level produces a practical monitoring/advisory action.
- **Existing backend:** SQLite stores readings and alerts; the rainfall simulator still works as before.

## Run

Windows:

1. Open the `landslide_backend` folder.
2. Double-click `start.bat`.
3. Open `http://localhost:8000`.

Or from a terminal:

```bash
cd landslide_backend
pip install -r requirements.txt
python3 main.py
```

Then open `http://localhost:8000`.
Interactive API docs (Swagger UI) are auto-generated at `http://localhost:8000/docs`.

## API

| Method | Path                | Auth required | Description                                   |
|--------|---------------------|:---:|------------------------------------------------|
| GET    | `/`                 |    | Dashboard (static/index.html)                  |
| GET    | `/api/health`       |    | Health check + model/DB info                    |
| GET    | `/api/state`        |    | Full state: stations, readings, alerts, tick    |
| POST   | `/api/simulate/step`|    | Advance the rainfall simulation by one tick     |
| POST   | `/api/reset`        | ✅  | Reset readings/alerts back to baseline          |
| POST   | `/api/telemetry`    | ✅  | Ingest a real sensor/weather-feed reading       |

All responses are validated against Pydantic models (`Station`, `Alert`, `StateResponse`, etc. in `main.py`), so malformed data is caught before it ever reaches the client.

### Feeding it real sensor data

`POST /api/telemetry` is the integration point for a real rain gauge, soil-moisture probe, or third-party weather API — it's separate from the built-in simulator and requires an API key:

```bash
curl -X POST http://localhost:8000/api/telemetry \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-change-me" \
  -d '{"station_id": "nongpoh", "rain72": 180, "intensity": 22, "moisture": 88}'
```

The default key is `demo-key-change-me`, printed to the console every time the server starts. Set a real one before deploying anywhere public:

```bash
# macOS / Linux
export SLOPESENSE_API_KEY="your-real-key"
# Windows (PowerShell)
$env:SLOPESENSE_API_KEY="your-real-key"
```

If the reading pushes a station's risk into High or Critical, `notifications.py` logs the alert to the console and — if `ALERT_WEBHOOK_URL` is set — also POSTs it to that URL, so you can wire it into a real Slack/Discord channel for testing without touching the code.

### Evaluating the model

```bash
python evaluate_model.py
```

Trains the same logistic regression on synthetic data and reports accuracy, precision, recall, F1, and ROC-AUC on a held-out synthetic test set. This confirms the training loop works — it does **not** validate real-world landslide prediction accuracy, since the labels are synthetic (see "Important ML note" below).

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest test_app.py
```

## Important ML note

The included model is a **demonstration ML model** trained from synthetic data so the project remains runnable without downloading packages or a dataset. It is not suitable for real emergency decisions.

For a real deployment, replace the synthetic training data in `ml_model.py` with labelled historical observations containing rainfall, soil moisture, slope movement/tilt, terrain/geology and a confirmed landslide outcome. A real project can then evaluate the model with train/test splits, precision, recall, F1, ROC-AUC and calibration.

## Suggested real-data upgrade

A future CSV can contain columns such as:

`rain24, rain72, intensity, moisture, slope_angle, elevation, drainage, history, tilt_rate, forecast6h, landslide`

Then train the model from those observations instead of synthetic examples.

## Prototype vs production

**Real in this prototype:** HTTP server, API, SQLite database, persistence, risk calculation, ML inference, map UI, explainability and alert logic.

**Simulated:** rainfall/soil/tilt observations and the labels used to train the demonstration ML model.

For operational use, connect trusted weather/rainfall feeds and physical sensors, validate the ML model against historical events, and have qualified geotechnical authorities define warning thresholds and response procedures.
