from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from backend.database.connection import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Queue(Base):
    __tablename__ = "queues"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), default="")
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=_utcnow)

    counters = relationship("Counter", back_populates="queue", cascade="all, delete-orphan")
    observations = relationship("QueueObservation", back_populates="queue")
    service_events = relationship("ServiceEvent", back_populates="queue")
    waiting_events = relationship("WaitingEvent", back_populates="queue")
    predictions = relationship("Prediction", back_populates="queue")
    anomalies = relationship("Anomaly", back_populates="queue")


class Counter(Base):
    __tablename__ = "counters"

    id = Column(Integer, primary_key=True, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    name = Column(String(100), nullable=False)
    active = Column(Boolean, default=True)

    queue = relationship("Queue", back_populates="counters")
    service_events = relationship("ServiceEvent", back_populates="counter")


class CameraConfig(Base):
    __tablename__ = "camera_configs"

    id = Column(Integer, primary_key=True, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    source = Column(String(200), default="webcam")
    resolution_width = Column(Integer, default=640)
    resolution_height = Column(Integer, default=480)
    fps = Column(Integer, default=30)
    roi = Column(JSON, default=dict)  # Region of Interest coordinates


class QueueObservation(Base):
    __tablename__ = "queue_observations"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=_utcnow, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    counter_id = Column(Integer, ForeignKey("counters.id"), nullable=True)
    queue_length = Column(Integer, default=0)
    arrival_rate = Column(Float, default=0.0)
    departure_rate = Column(Float, default=0.0)
    active_counters = Column(Integer, default=1)

    queue = relationship("Queue", back_populates="observations")


class ServiceEvent(Base):
    __tablename__ = "service_events"

    id = Column(Integer, primary_key=True, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    counter_id = Column(Integer, ForeignKey("counters.id"), nullable=True)
    person_temp_id = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    service_time = Column(Float, nullable=True)  # seconds

    queue = relationship("Queue", back_populates="service_events")
    counter = relationship("Counter", back_populates="service_events")


class WaitingEvent(Base):
    __tablename__ = "waiting_events"

    id = Column(Integer, primary_key=True, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    person_temp_id = Column(String(50), nullable=False)
    queue_enter_time = Column(DateTime, nullable=False)
    service_start_time = Column(DateTime, nullable=True)
    actual_wait = Column(Float, nullable=True)  # seconds

    queue = relationship("Queue", back_populates="waiting_events")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=_utcnow, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    predicted_wait = Column(Float, nullable=False)  # seconds
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)

    queue = relationship("Queue", back_populates="predictions")


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=_utcnow, index=True)
    queue_id = Column(Integer, ForeignKey("queues.id"), nullable=False)
    type = Column(String(100), nullable=False)
    severity = Column(String(50), default="medium")
    evidence = Column(Text, default="")
    resolved = Column(Boolean, default=False)

    queue = relationship("Queue", back_populates="anomalies")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    train_date = Column(DateTime, default=_utcnow)
    MAE = Column(Float, nullable=True)
    RMSE = Column(Float, nullable=True)
    R2 = Column(Float, nullable=True)
    dataset_version = Column(String(50), default="v1")
