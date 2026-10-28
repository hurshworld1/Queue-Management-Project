import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base, get_db
from backend.main import app

# Test database setup — use a separate in-memory-ish SQLite
TEST_DATABASE_URL = "sqlite:///./test_queue.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)


def admin_headers():
    """Log in with default credentials and return auth headers."""
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200, res.text
    token = res.json()["token"]
    return {"Authorization": f"Bearer {token}"}


class TestAdminAuth:
    def test_login_success(self):
        res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert len(data["token"]) >= 32

    def test_login_wrong_password(self):
        res = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
        assert res.status_code == 401

    def test_login_wrong_username(self):
        res = client.post("/api/auth/login", json={"username": "nobody", "password": "admin123"})
        assert res.status_code == 401

    def test_protected_endpoint_rejects_no_token(self):
        res = client.post("/api/camera/stop")
        assert res.status_code == 401

    def test_protected_endpoint_rejects_bad_token(self):
        res = client.post("/api/camera/stop", headers={"Authorization": "Bearer bogus"})
        assert res.status_code == 401

    def test_camera_frame_requires_auth(self):
        res = client.get("/api/camera/frame")
        assert res.status_code == 401

    def test_logout_invalidates_token(self):
        res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        token = res.json()["token"]
        hdrs = {"Authorization": f"Bearer {token}"}
        assert client.post("/api/camera/stop", headers=hdrs).status_code == 200
        client.post("/api/auth/logout", headers=hdrs)
        assert client.post("/api/camera/stop", headers=hdrs).status_code == 401


class TestQueueAPI:
    def test_list_queues_empty(self):
        response = client.get("/api/queues")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_queue(self):
        response = client.post("/api/queues", json={"name": "Canteen Counter 1", "location": "Building A"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Canteen Counter 1"
        assert data["location"] == "Building A"
        assert data["id"] is not None

    def test_list_queues_after_create(self):
        client.post("/api/queues", json={"name": "Queue 1"})
        client.post("/api/queues", json={"name": "Queue 2"})
        response = client.get("/api/queues")
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_get_queue(self):
        create_res = client.post("/api/queues", json={"name": "Test Queue"})
        queue_id = create_res.json()["id"]
        response = client.get(f"/api/queues/{queue_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Test Queue"

    def test_get_queue_not_found(self):
        response = client.get("/api/queues/999")
        assert response.status_code == 404


class TestLiveStatus:
    def test_live_status(self):
        create_res = client.post("/api/queues", json={"name": "Live Queue"})
        queue_id = create_res.json()["id"]
        response = client.get(f"/api/queues/{queue_id}/live")
        assert response.status_code == 200
        data = response.json()
        assert data["queue_id"] == queue_id
        assert "estimated_wait_minutes" in data


class TestEvents:
    def test_store_event(self):
        create_res = client.post("/api/queues", json={"name": "Event Queue"})
        queue_id = create_res.json()["id"]
        response = client.post("/api/events", json={
            "queue_id": queue_id,
            "queue_length": 5,
            "arrival_rate": 1.5,
            "departure_rate": 0.8,
            "active_counters": 2,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_store_service_event(self):
        create_res = client.post("/api/queues", json={"name": "Service Queue"})
        queue_id = create_res.json()["id"]
        response = client.post("/api/service-events", json={
            "queue_id": queue_id,
            "person_temp_id": "person_0",
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T10:00:30",
            "service_time": 30.0,
        })
        assert response.status_code == 200


class TestAnalytics:
    def test_get_analytics(self):
        create_res = client.post("/api/queues", json={"name": "Analytics Queue"})
        queue_id = create_res.json()["id"]
        response = client.get(f"/api/queues/{queue_id}/analytics")
        assert response.status_code == 200


class TestConfig:
    def test_create_counter(self):
        # config endpoints are admin-only; queue creation stays public for the pipeline
        create_res = client.post("/api/queues", json={"name": "Counter Queue"})
        queue_id = create_res.json()["id"]
        response = client.post("/api/config/counter", json={
            "queue_id": queue_id,
            "name": "Counter 1",
        }, headers=admin_headers())
        assert response.status_code == 200
        assert response.json()["name"] == "Counter 1"


class TestModelMetrics:
    def test_model_metrics_returns_response(self):
        response = client.get("/api/model/metrics")
        assert response.status_code == 200
        data = response.json()
        # Returns either online predictor metrics or batch model info
        has_online = "linear_weight" in data or "total_updates" in data
        has_batch = "model_name" in data or "error" in data
        assert has_online or has_batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
