from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool
import os

# Use relative path so the DB lives in the project root
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATABASE_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(_DATABASE_DIR, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(_DATABASE_DIR, 'queue_management.db')}",
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from backend.database.models import (
        Queue,
        Counter,
        CameraConfig,
        QueueObservation,
        ServiceEvent,
        WaitingEvent,
        Prediction,
        Anomaly,
        ModelRun,
    )
    Base.metadata.create_all(bind=engine)
