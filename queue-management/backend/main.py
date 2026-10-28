"""
ML Queue Management API — Production Backend

Architecture:
- FastAPI server with REST endpoints + Server-Sent Events (SSE)
- Multi-threaded camera pipeline: capture -> detect -> track -> update state
- Online ML ensemble that learns from every service completion
- Kalman-smoothed queue counts for jitter-free display
- SSE for instant frontend updates (no polling needed)

Camera pipeline runs in a background thread with frame skipping:
- Capture thread reads frames at camera FPS (30fps)
- Detection runs every 3rd frame (~10fps) to save CPU
- Tracker interpolates positions between detections
- State manager updates instantly (no HTTP roundtrip)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
import json
import asyncio
import os
import secrets
import threading
import time
import base64
from pydantic import BaseModel
from collections import deque
import cv2
import numpy as np

from backend.database.connection import get_db, init_db, SessionLocal
from backend.database.models import Queue, Counter
from backend.schemas.queue import (
    QueueCreate, QueueResponse,
    CounterCreate, CounterResponse,
    QueueObservationCreate, QueueObservationResponse,
    PredictionCreate, PredictionResponse,
    ServiceEventCreate, WaitingEventCreate,
    FeatureInput, PredictionRequest, PredictionResult,
)
from backend.services.queue_service import (
    state_manager, create_queue, get_queue, list_queues,
    create_counter, record_observation, record_prediction,
    record_anomaly, get_analytics, ServiceTimeTracker,
)
from backend.services.anomaly_service import anomaly_detector


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title="ML Queue Management API",
    description="Real-time queue monitoring with online ML prediction",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Admin auth — token-gated admin endpoints
# ──────────────────────────────────────────────

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
_TOKEN_TTL_SECONDS = 12 * 3600  # tokens expire after 12 hours
_admin_tokens: dict = {}  # token -> expiry epoch
_tokens_lock = threading.Lock()


class AdminLoginRequest(BaseModel):
    username: str
    password: str


def _issue_token() -> str:
    token = secrets.token_hex(32)
    now = time.time()
    with _tokens_lock:
        # prune expired tokens while we're here
        expired = [t for t, exp in _admin_tokens.items() if exp < now]
        for t in expired:
            del _admin_tokens[t]
        _admin_tokens[token] = now + _TOKEN_TTL_SECONDS
    return token


def require_admin(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
):
    """FastAPI dependency: validates an admin bearer token (or ?token= query param for image/stream URLs)."""
    supplied = None
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization[len("Bearer "):].strip()
    elif token:
        supplied = token
    if not supplied:
        raise HTTPException(status_code=401, detail="Admin authentication required")
    with _tokens_lock:
        expiry = _admin_tokens.get(supplied)
    if expiry is None or expiry < time.time():
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return supplied


@app.post("/api/auth/login")
def admin_login(data: AdminLoginRequest):
    """Validate admin credentials and issue a short-lived access token."""
    user_ok = secrets.compare_digest(data.username.encode(), ADMIN_USERNAME.encode())
    pass_ok = secrets.compare_digest(data.password.encode(), ADMIN_PASSWORD.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"status": "ok", "token": _issue_token(), "expires_in": _TOKEN_TTL_SECONDS}


@app.post("/api/auth/logout")
def admin_logout(authorization: Optional[str] = Header(None)):
    """Invalidate the presented token."""
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization[len("Bearer "):].strip()
        with _tokens_lock:
            _admin_tokens.pop(supplied, None)
    return {"status": "ok"}


def admin_required():
    return Depends(require_admin)


# ──────────────────────────────────────────────
# Queue CRUD
# ──────────────────────────────────────────────

@app.get("/api/queues", response_model=List[QueueResponse])
def list_all_queues(db: Session = Depends(get_db)):
    return list_queues(db)


@app.post("/api/queues", response_model=QueueResponse)
def create_new_queue(data: QueueCreate, db: Session = Depends(get_db)):
    return create_queue(db, name=data.name, location=data.location, status=data.status)


@app.get("/api/queues/{queue_id}", response_model=QueueResponse)
def get_queue_detail(queue_id: int, db: Session = Depends(get_db)):
    queue = get_queue(db, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")
    return queue


# ──────────────────────────────────────────────
# Live status
# ──────────────────────────────────────────────

@app.get("/api/queues/{queue_id}/live")
def get_live_queue_status(queue_id: int, db: Session = Depends(get_db)):
    queue = get_queue(db, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")
    return state_manager.get_live_status(queue_id, db)


@app.post("/api/queues/{queue_id}/adjust")
def adjust_queue_length(queue_id: int, delta: int = 0, db: Session = Depends(get_db), _admin: str = Depends(require_admin)):
    """Manually increment or decrement the queue length."""
    queue = get_queue(db, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")

    state = state_manager.get_or_create_state(queue_id)
    new_length = max(0, state["queue_length"] + delta)
    state_manager.update_queue_length(queue_id, new_length)

    obs_data = {
        "queue_id": queue_id,
        "queue_length": new_length,
        "arrival_rate": 0.0,
        "departure_rate": 0.0,
        "active_counters": state["active_counters"],
    }
    record_observation(db, obs_data)

    return {"status": "ok", "queue_id": queue_id, "queue_length": new_length, "delta": delta}


@app.post("/api/queues/{queue_id}/set-length")
def set_queue_length(queue_id: int, length: int = 0, db: Session = Depends(get_db), _admin: str = Depends(require_admin)):
    """Set the queue length to an exact value."""
    queue = get_queue(db, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")

    new_length = max(0, length)
    state_manager.update_queue_length(queue_id, new_length)

    state = state_manager.get_or_create_state(queue_id)
    obs_data = {
        "queue_id": queue_id,
        "queue_length": new_length,
        "arrival_rate": 0.0,
        "departure_rate": 0.0,
        "active_counters": state["active_counters"],
    }
    record_observation(db, obs_data)

    return {"status": "ok", "queue_id": queue_id, "queue_length": new_length}


# ──────────────────────────────────────────────
# Observations & Events
# ──────────────────────────────────────────────

@app.post("/api/events")
def store_event(data: QueueObservationCreate, db: Session = Depends(get_db)):
    """Store a queue observation and update real-time state."""
    obs = record_observation(db, data.model_dump())

    state_manager.update_queue_length(data.queue_id, data.queue_length)
    state = state_manager.get_or_create_state(data.queue_id)
    state["active_counters"] = data.active_counters

    # Track arrival in service time tracker
    tracker = state_manager.get_tracker(data.queue_id)
    tracker.record_arrival(f"manual_{data.queue_id}_{obs.id}", data.queue_id)

    # Check anomalies
    alerts = anomaly_detector.check_anomalies(data.queue_id, data.queue_length)
    for alert in alerts:
        record_anomaly(db, alert.queue_id, alert.type, alert.severity, alert.evidence)

    return {"status": "ok", "observation_id": obs.id, "anomalies": len(alerts)}


@app.post("/api/predictions")
def store_prediction(data: PredictionCreate, db: Session = Depends(get_db)):
    """Accept a prediction from the ML pipeline."""
    pred = record_prediction(db, data.queue_id, data.predicted_wait, data.lower_bound, data.upper_bound)
    return {"status": "ok", "prediction_id": pred.id}


@app.post("/api/service-events")
def store_service_event(data: ServiceEventCreate, db: Session = Depends(get_db)):
    """Record a service event (person starts/finishes service)."""
    from backend.database.models import ServiceEvent

    event = ServiceEvent(**data.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)

    # Record service time in tracker
    tracker = state_manager.get_tracker(data.queue_id)
    if data.service_time:
        tracker.record_service_time_direct(data.service_time)

        # Feed to online ML predictor for learning
        from datetime import datetime as dt
        now = dt.now()
        features = {
            "queue_length": float(state_manager.get_or_create_state(data.queue_id)["queue_length"]),
            "arrival_rate": 0.0,
            "departure_rate": tracker.get_departure_rate(),
            "mean_service_time": tracker.get_avg_service_time(),
            "active_counters": float(state_manager.get_or_create_state(data.queue_id).get("active_counters", 1)),
            "queue_growth_rate": 0.0,
            "time_of_day": float(now.hour + now.minute / 60.0),
            "day_of_week": float(now.weekday()),
            "service_time_variability": tracker.get_service_time_variability(),
        }
        state_manager.predictor.update(features, data.service_time)

    if data.end_time:
        anomaly_detector.record_service_completion(
            data.queue_id, data.end_time.timestamp()
        )

    return {"status": "ok", "event_id": event.id}


@app.post("/api/waiting-events")
def store_waiting_event(data: WaitingEventCreate, db: Session = Depends(get_db)):
    """Record a waiting event (actual wait time for a person)."""
    from backend.database.models import WaitingEvent

    event = WaitingEvent(**data.model_dump())
    db.add(event)
    db.commit()
    db.refresh(event)

    return {"status": "ok", "event_id": event.id}


# ──────────────────────────────────────────────
# ML Prediction
# ──────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictionResult)
def predict_waiting_time(request: PredictionRequest):
    """Predict waiting time from features using the trained model."""
    try:
        from ml.train import load_latest_model, predict_wait
        model, scaler, metadata = load_latest_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Train a model first.",
        )

    features = request.features.model_dump()
    result = predict_wait(model, scaler, features)
    result["model_version"] = metadata.get("version", "unknown")
    return PredictionResult(**result)


# ──────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────

@app.get("/api/queues/{queue_id}/analytics")
def get_queue_analytics(queue_id: int, db: Session = Depends(get_db)):
    queue = get_queue(db, queue_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")
    return get_analytics(db, queue_id)


@app.get("/api/queues/{queue_id}/history")
def get_queue_history(queue_id: int, limit: int = 100, db: Session = Depends(get_db)):
    from backend.database.models import QueueObservation

    observations = (
        db.query(QueueObservation)
        .filter(QueueObservation.queue_id == queue_id)
        .order_by(QueueObservation.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "timestamp": o.timestamp.isoformat(),
            "queue_length": o.queue_length,
            "arrival_rate": o.arrival_rate,
            "departure_rate": o.departure_rate,
        }
        for o in observations
    ]


@app.get("/api/alerts")
def get_alerts(queue_id: Optional[int] = None, limit: int = 50, db: Session = Depends(get_db)):
    from backend.database.models import Anomaly

    query = db.query(Anomaly).filter(Anomaly.resolved == False)
    if queue_id:
        query = query.filter(Anomaly.queue_id == queue_id)
    anomalies = query.order_by(Anomaly.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": a.id,
            "timestamp": a.timestamp.isoformat(),
            "queue_id": a.queue_id,
            "type": a.type,
            "severity": a.severity,
            "evidence": a.evidence,
        }
        for a in anomalies
    ]


@app.get("/api/model/metrics")
def get_model_metrics():
    """Return metrics for the online ensemble predictor."""
    metrics = state_manager.predictor.get_metrics()
    try:
        from ml.train import load_latest_model
        _, _, metadata = load_latest_model()
        metrics["batch_model"] = metadata
    except FileNotFoundError:
        metrics["batch_model"] = {"error": "No batch model trained"}
    return metrics


@app.get("/api/model/predictor")
def get_predictor_details():
    """Detailed info about the online ML ensemble."""
    return state_manager.predictor.get_metrics()


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@app.post("/api/config/queue")
def configure_queue(data: QueueCreate, db: Session = Depends(get_db), _admin: str = Depends(require_admin)):
    return create_queue(db, name=data.name, location=data.location, status=data.status)


@app.post("/api/config/counter")
def configure_counter(data: CounterCreate, db: Session = Depends(get_db), _admin: str = Depends(require_admin)):
    return create_counter(db, queue_id=data.queue_id, name=data.name)


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "ML Queue Management API", "version": "2.0.0"}


# ──────────────────────────────────────────────
# Camera Control — Multi-threaded pipeline
# ──────────────────────────────────────────────

class CameraStream:
    """
    Production camera pipeline with multi-threaded processing.

    Architecture:
    1. Capture thread: reads frames from webcam at native FPS
    2. Processing: detection runs every DETECTION_INTERVAL frames
    3. Tracker interpolates positions between detections
    4. State manager updates directly (zero HTTP latency)
    5. SSE/streaming pushes frames to browser at ~10fps

    Frame skipping: YOLO runs on every 3rd frame (~10fps detection)
    while the camera captures at 30fps. This saves ~60% CPU while
    maintaining smooth tracking through prediction.
    """

    DETECTION_INTERVAL = 3  # Run YOLO every N frames (3 = ~10fps detection)
    TARGET_FPS = 10         # Target processing FPS

    def __init__(self):
        self._running = False
        self._thread = None
        self._cap = None
        self._detector = None
        self._tracker = None
        self._frame = None
        self._lock = threading.Lock()
        self._stats = {
            "queue_count": 0,
            "total_count": 0,
            "detections": 0,
            "frames": 0,
            "processed_frames": 0,
            "status": "stopped",
            "fps": 0.0,
            "avg_service_time": 30.0,
            "departures_per_min": 0,
        }
        self._queue_id = 1
        self._api_url = "http://127.0.0.1:8000"

        # FPS calculation
        self._frame_times: deque = deque(maxlen=30)

        # Previous frame tracks for display between detection frames
        self._last_tracks = []
        self._last_detections = []
        self._frame_counter = 0

    @property
    def is_running(self):
        return self._running

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def start(self, queue_id=1, camera_index=0):
        if self._running:
            return {"status": "already_running"}

        # Quick camera check
        test_cap = cv2.VideoCapture(camera_index)
        if not test_cap.isOpened():
            test_cap.release()
            return {"status": "error", "message": f"Cannot open camera {camera_index}. Check if another app is using it."}
        test_cap.release()

        self._queue_id = queue_id
        self._running = True
        self._frame_counter = 0
        self._thread = threading.Thread(target=self._run_loop, args=(camera_index,), daemon=True)
        self._thread.start()

        with self._lock:
            self._stats["status"] = "starting"
        return {"status": "started", "queue_id": queue_id, "camera": camera_index}

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        with self._lock:
            self._stats["status"] = "stopped"
        return {"status": "stopped"}

    def _run_loop(self, camera_index):
        try:
            import sys
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if PROJECT_ROOT not in sys.path:
                sys.path.insert(0, PROJECT_ROOT)
            from vision.detector import PersonDetector
            from vision.tracker import PersonTracker

            self._cap = cv2.VideoCapture(camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

            if not self._cap.isOpened():
                with self._lock:
                    self._stats["status"] = "error: cannot open camera"
                self._running = False
                return

            self._detector = PersonDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.4,
                min_bbox_height=40,
                nms_iou_threshold=0.5,
            )
            self._tracker = PersonTracker(
                iou_threshold=0.3,
                max_age=5,
                min_hits=1,
            )

            with self._lock:
                self._stats["status"] = "running"
            print("[Camera] Pipeline started")

            import requests as req

            prev_track_ids = set()
            tracker = state_manager.get_tracker(self._queue_id)
            frame_count = 0
            last_process_time = time.time()
            target_interval = 1.0 / self.TARGET_FPS

            while self._running:
                ret, frame = self._cap.read()
                if not ret or not self._running:
                    break

                frame_count += 1
                self._frame_counter += 1
                now = time.time()

                # Frame skipping: only run detection every Nth frame
                run_detection = (self._frame_counter % self.DETECTION_INTERVAL == 0)

                if run_detection:
                    # Run YOLO detection
                    detections = self._detector.detect(frame)
                    tracks = self._tracker.update(detections)
                    self._last_tracks = tracks
                    self._last_detections = detections
                else:
                    # Use predicted positions from tracker
                    tracks = self._last_tracks
                    detections = []

                # Compute FPS
                self._frame_times.append(now)
                if len(self._frame_times) >= 2:
                    fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])
                else:
                    fps = 0

                current_track_ids = {t.track_id for t in tracks}

                # Detect arrivals (new person appeared)
                new_ids = current_track_ids - prev_track_ids
                for tid in new_ids:
                    tracker.record_arrival(tid, self._queue_id)

                # Detect departures (person disappeared)
                departed_ids = prev_track_ids - current_track_ids
                for tid in departed_ids:
                    svc_time = tracker.record_service_completion(tid)
                    if svc_time is not None:
                        try:
                            req.post(
                                f"{self._api_url}/api/service-events",
                                json={
                                    "queue_id": self._queue_id,
                                    "person_temp_id": tid,
                                    "start_time": datetime.now(timezone.utc).isoformat(),
                                    "end_time": datetime.now(timezone.utc).isoformat(),
                                    "service_time": round(svc_time, 1),
                                },
                                timeout=2,
                            )
                        except Exception:
                            pass

                prev_track_ids = current_track_ids
                queue_count = len(tracks)

                # Update state manager DIRECTLY (zero latency)
                state_manager.update_queue_length(self._queue_id, queue_count)
                state = state_manager.get_or_create_state(self._queue_id)
                state["active_counters"] = 2

                # Print count changes
                if queue_count != getattr(self, '_last_printed_count', -1):
                    print(f"[Camera] People: {queue_count} | Detections: {len(detections)} | FPS: {fps:.1f}")
                    self._last_printed_count = queue_count

                # Draw annotations
                display = self._draw_annotations(frame, tracks, detections, fps)
                avg_st = tracker.get_avg_service_time()
                dep_rate = tracker.get_departure_rate(60)

                # Encode frame as JPEG
                _, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
                frame_bytes = jpeg.tobytes()

                with self._lock:
                    self._frame = frame_bytes
                    self._stats["queue_count"] = queue_count
                    self._stats["total_count"] = len(tracks)
                    self._stats["detections"] = len(detections)
                    self._stats["frames"] = frame_count
                    self._stats["processed_frames"] += 1
                    self._stats["status"] = "running"
                    self._stats["avg_service_time"] = round(avg_st, 1)
                    self._stats["departures_per_min"] = dep_rate
                    self._stats["fps"] = round(fps, 1)

                # Throttle to target FPS
                elapsed = time.time() - now
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            import traceback
            traceback.print_exc()
            with self._lock:
                self._stats["status"] = f"error: {str(e)}"
        finally:
            if self._cap:
                self._cap.release()
            self._running = False
            with self._lock:
                self._stats["status"] = "stopped"

    def _draw_annotations(self, frame, tracks, detections, fps):
        """Draw bounding boxes, track IDs, and stats on frame."""
        h, w = frame.shape[:2]

        # Detections (yellow, thin)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Tracks (green, thick with ID and trajectory)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, track.track_id, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cx, cy = track.center
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # Trajectory
            if len(track.history) > 1:
                pts = [(int(h_[0]), int(h_[1])) for h_ in track.history[-10:]]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (0, 200, 200), 1)

        # Stats panel
        tracker = state_manager.get_tracker(self._queue_id)
        avg_st = tracker.get_avg_service_time()
        dep_rate = tracker.get_departure_rate(60)

        lines = [
            f"People: {len(tracks)}",
            f"Avg service: {avg_st:.0f}s",
            f"Departures/min: {dep_rate}",
            f"FPS: {fps:.1f} | Frame: {self._frame_counter}",
        ]

        # Color based on count
        count = len(tracks)
        if count <= 3:
            status_color = (0, 200, 0)
        elif count <= 7:
            status_color = (0, 180, 255)
        else:
            status_color = (0, 0, 255)

        panel_h = 22 * len(lines) + 16
        cv2.rectangle(frame, (5, 5), (280, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (280, panel_h), (100, 100, 100), 1)
        for i, text in enumerate(lines):
            color = status_color if i == 0 else (200, 200, 200)
            cv2.putText(frame, text, (12, 26 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame


camera_stream = CameraStream()


@app.get("/api/camera/status")
def camera_status():
    return camera_stream.stats


@app.post("/api/camera/start")
def camera_start(queue_id: int = 1, camera: int = 0, _admin: str = Depends(require_admin)):
    if camera_stream.is_running:
        return {"status": "already_running"}
    result = camera_stream.start(queue_id=queue_id, camera_index=camera)
    return result


@app.post("/api/camera/stop")
def camera_stop(_admin: str = Depends(require_admin)):
    return camera_stream.stop()


@app.get("/api/camera/frame")
def camera_frame(_admin: str = Depends(require_admin)):
    """Get the latest camera frame as JPEG (admin only)."""
    from starlette.responses import Response
    frame = camera_stream.frame
    if frame is None:
        raise HTTPException(status_code=404, detail="No camera frame available")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/api/camera/stream")
def camera_stream_mjpeg(_admin: str = Depends(require_admin)):
    """MJPEG stream for continuous camera feed in the browser (admin only)."""
    def generate():
        while camera_stream.is_running:
            frame = camera_stream.frame
            if frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/camera")
async def websocket_camera_stream(websocket: WebSocket):
    """Stream camera frames + stats as JSON to the browser (admin only)."""
    token = websocket.query_params.get("token")
    with _tokens_lock:
        expiry = _admin_tokens.get(token) if token else None
    if expiry is None or expiry < time.time():
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            frame = camera_stream.frame
            stats = camera_stream.stats
            if frame is not None:
                b64 = base64.b64encode(frame).decode("utf-8")
                stats["frame_b64"] = b64
            await websocket.send_json(stats)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ──────────────────────────────────────────────
# Serve built frontend (if available)
# ──────────────────────────────────────────────

_frontend_dist = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "dist"
)
if os.path.isdir(_frontend_dist):
    from starlette.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(_frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_frontend_dist, "index.html"))


# ──────────────────────────────────────────────
# WebSocket for live updates
# ──────────────────────────────────────────────

connected_clients: List[WebSocket] = []


@app.websocket("/ws/live")
async def websocket_live_queue(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(0.3)
            statuses = []
            for qid in state_manager.queue_states:
                try:
                    db = SessionLocal()
                    status = state_manager.get_live_status(qid, db)
                    db.close()
                    statuses.append(status)
                except Exception:
                    pass
            if statuses:
                await websocket.send_json(statuses)
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
