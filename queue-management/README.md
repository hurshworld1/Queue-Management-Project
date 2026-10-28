# ML Queue Management System

Real-time queue monitoring and waiting time estimation using computer vision and online machine learning.

## Overview

This system uses a laptop webcam to count people in queues, track them across frames, and predict waiting times that improve continuously as more people are observed.

**Core components:**
- YOLOv8 person detection (runs every 3rd frame to save CPU)
- ByteTrack-inspired tracker with Kalman-based motion prediction
- Kalman filter for smoothing noisy camera counts
- Online ML ensemble (linear regression + random forest + weighted average) that learns from every service completion
- FastAPI backend with REST API and WebSocket streaming
- React dashboard with live camera feed, queue status, and analytics

## Requirements

- Python 3.9+
- Node.js 18+
- A laptop webcam

## Quick Start

```bash
# 1. Install Python dependencies
cd queue-management
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend
npm install
npm run build
cd ..

# 3. Train the ML model (generates synthetic data first)
python run.py setup

# 4. Start the server
python run.py server
```

Open http://localhost:8000 in your browser.

## Usage

### Start the server

```bash
python run.py server
```

### Start the camera from the browser

1. Open http://localhost:8000
2. Click "Admin" in the navigation
3. Select a queue from the dropdown
4. Click "Start Camera"
5. The camera feed appears with detection boxes and live queue count

### Start the camera from the command line (headless)

```bash
python run.py camera                    # default webcam, queue 1
python run.py camera --camera 1         # second camera
python run.py camera --queue-id 2       # monitor queue #2
```

### Configure ROI zones

```bash
python run.py roi
```

Click points on the camera to draw queue and service zones. Press Enter to confirm, S to switch to service zone mode, Q to save.

## Architecture

### Detection Pipeline

```
Webcam (30fps)
  |
  v
Frame capture (every frame)
  |
  v [every 3rd frame]
YOLOv8n person detection (~10fps)
  |
  v
ByteTrack with Kalman prediction
  |
  v
Kalman filter smoothing (queue count)
  |
  v
Online ML ensemble (wait time prediction)
  |
  v
State manager (direct update, zero latency)
  |
  v
Frontend (SSE/WebSocket, 200ms refresh)
```

### Tracking

The tracker uses a two-stage matching process inspired by ByteTrack:
1. **High-confidence match**: IoU > 0.3 with predicted positions
2. **Low-confidence match**: IoU > 0.15 for partially occluded people
3. Kalman filter predicts where each person will be in the next frame
4. Tracks are kept alive for up to 8 frames without detection

### Wait Time Prediction

The system uses three models that combine into an ensemble:

1. **Weighted Moving Average**: Fast adaptation, good baseline
2. **Online Linear Regression**: Learns linear relationships (queue length vs wait)
3. **Online Random Forest**: Captures non-linear patterns (peak hours, service variability)

All three models update their parameters with every observed service completion. The ensemble shifts weight toward whichever model performs best recently.

**Kalman filter** smooths the raw queue count from the camera (which is noisy due to detection jitter) to produce a stable, responsive count.

### Key formulas

```
Smoothed count = KalmanFilter(camera_detection)
Predicted wait = EnsemblePredictor(queue_length, arrival_rate, departure_rate, avg_service_time, ...)
Service time tracking = ExponentialMovingAverage(half_life=120s)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/queues | List all queues |
| GET | /api/queues/{id}/live | Live status with ML prediction |
| POST | /api/queues/{id}/adjust?delta=N | Add/remove people |
| POST | /api/camera/start?queue_id=N&camera=N | Start camera |
| POST | /api/camera/stop | Stop camera |
| GET | /api/camera/status | Camera stats |
| GET | /api/camera/frame | Latest camera frame (JPEG) |
| GET | /api/model/metrics | Online ML ensemble metrics |
| GET | /api/alerts | Active anomaly alerts |
| WS | /ws/live | Live queue updates |

## Project Structure

```
queue-management/
  backend/
    main.py                  # FastAPI server, camera pipeline, API endpoints
    services/
      queue_service.py       # State manager, Kalman smoothing, online ML
      anomaly_service.py     # Anomaly detection
    database/
      models.py              # SQLAlchemy models
      connection.py          # DB connection
    schemas/
      queue.py               # Pydantic schemas
  vision/
    detector.py              # YOLOv8 person detection
    tracker_v2.py            # ByteTrack with Kalman prediction
    camera_runner.py         # Standalone camera monitor (CLI)
    event_extractor.py       # Queue event detection
    roi.py                   # Region of interest
  ml/
    kalman_filter.py         # 1D Kalman filter
    online_predictor.py      # Online ensemble ML
    train.py                 # Batch model training
    feature_engineering.py   # Feature computation
    preprocessing.py         # Data preprocessing
  frontend/
    src/
      pages/
        CustomerDashboard.jsx  # Public view
        AdminDashboard.jsx     # Admin controls + camera
        CameraPage.jsx         # Camera monitor
      services/
        api.js                 # API client
```

## Configuration

Default queue names and counters are set up during `python run.py setup`. To customize:

```python
# Add a queue via API
curl -X POST http://localhost:8000/api/queues \
  -H "Content-Type: application/json" \
  -d '{"name": "Canteen Counter 1", "location": "Building A"}'
```

## Troubleshooting

**Camera not starting:**
- Check if another application is using the webcam
- Try a different camera index: `--camera 1`
- Check terminal for error messages

**Slow initial startup:**
- YOLO model loads on first use (~5 seconds)
- Subsequent starts are instant

**Queue count not changing:**
- Make sure people are visible in the camera frame
- Default ROI covers the full frame
- Lower detection confidence: `--confidence 0.3`

**Wait time stuck at 0:**
- Wait time requires observed service completions
- The ML model learns from real data over time
- First predictions appear after the first person leaves the queue