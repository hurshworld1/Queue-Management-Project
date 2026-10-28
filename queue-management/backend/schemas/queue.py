from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


# Queue schemas
class QueueCreate(BaseModel):
    name: str
    location: str = ""
    status: str = "active"


class QueueResponse(BaseModel):
    id: int
    name: str
    location: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# Counter schemas
class CounterCreate(BaseModel):
    name: str
    queue_id: int
    active: bool = True


class CounterResponse(BaseModel):
    id: int
    queue_id: int
    name: str
    active: bool

    model_config = {"from_attributes": True}


# Live queue status
class LiveQueueStatus(BaseModel):
    queue_id: int
    queue_name: str
    queue_length: int
    estimated_wait_minutes: float
    lower_bound_minutes: Optional[float] = None
    upper_bound_minutes: Optional[float] = None
    trend: str = "stable"  # growing, shrinking, stable
    status: str = "low"  # low, moderate, high
    last_updated: datetime
    active_counters: int


# Queue observation
class QueueObservationCreate(BaseModel):
    queue_id: int
    counter_id: Optional[int] = None
    queue_length: int = 0
    arrival_rate: float = 0.0
    departure_rate: float = 0.0
    active_counters: int = 1


class QueueObservationResponse(BaseModel):
    id: int
    timestamp: datetime
    queue_id: int
    queue_length: int
    arrival_rate: float
    departure_rate: float
    active_counters: int

    model_config = {"from_attributes": True}


# Prediction schemas
class PredictionCreate(BaseModel):
    timestamp: datetime
    queue_id: int
    predicted_wait: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class PredictionResponse(BaseModel):
    id: int
    timestamp: datetime
    queue_id: int
    predicted_wait: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]

    model_config = {"from_attributes": True}


# Event schemas
class ServiceEventCreate(BaseModel):
    queue_id: int
    counter_id: Optional[int] = None
    person_temp_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    service_time: Optional[float] = None


class WaitingEventCreate(BaseModel):
    queue_id: int
    person_temp_id: str
    queue_enter_time: datetime
    service_start_time: Optional[datetime] = None
    actual_wait: Optional[float] = None


# Anomaly schemas
class AnomalyResponse(BaseModel):
    id: int
    timestamp: datetime
    queue_id: int
    type: str
    severity: str
    evidence: str
    resolved: bool

    model_config = {"from_attributes": True}


# Feature input for ML
class FeatureInput(BaseModel):
    queue_length: int
    arrival_rate: float
    departure_rate: float
    estimated_service_rate: float = 0.0
    mean_service_time: float = 0.0
    median_service_time: float = 0.0
    service_time_variability: float = 0.0
    active_counters: int = 1
    queue_growth_rate: float = 0.0
    recent_waiting_history: float = 0.0
    time_of_day: float = 0.0
    day_of_week: float = 0.0
    recent_anomaly_flag: bool = False


class PredictionRequest(BaseModel):
    queue_id: int
    features: FeatureInput


class PredictionResult(BaseModel):
    predicted_wait: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    model_version: str = ""


# Analytics schemas
class AnalyticsResponse(BaseModel):
    average_wait_minutes: float
    average_service_time_minutes: float
    peak_hours: List[dict]
    queue_trends: List[dict]
    prediction_accuracy: Optional[dict] = None


# Model metrics
class ModelMetricsResponse(BaseModel):
    model_name: str
    version: str
    MAE: float
    RMSE: float
    R2: float
    train_date: datetime
    dataset_version: str


# Camera config
class CameraConfigCreate(BaseModel):
    queue_id: int
    source: str = "webcam"
    resolution_width: int = 640
    resolution_height: int = 480
    fps: int = 30
    roi: dict = {}
