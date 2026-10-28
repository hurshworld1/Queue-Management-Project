#!/usr/bin/env python3
"""
ML Queue Management System

Usage:
    python run.py setup      - Initialize DB, generate data, train models
    python run.py server     - Start the FastAPI backend server
    python run.py train      - Regenerate data and retrain ML models
    python run.py test       - Run the test suite
    python run.py camera     - Start camera queue monitor (terminal mode)
    python run.py roi        - Interactive ROI zone setup for camera
"""
import sys
import os
import subprocess
import time

# Ensure we're in the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def cmd_setup():
    print("=" * 60)
    print("  Setting up ML Queue Management System")
    print("=" * 60)

    print("\n  [1/4] Initializing database...")
    from backend.database.connection import init_db, SessionLocal
    from backend.database.models import Queue, Counter
    from backend.services.queue_service import create_queue, create_counter

    init_db()
    db = SessionLocal()
    existing = db.query(Queue).count()

    if existing == 0:
        print("  Creating sample queues...")
        q1 = create_queue(db, name="Canteen Counter 1", location="Main Building")
        q2 = create_queue(db, name="Canteen Counter 2", location="Main Building")
        q3 = create_queue(db, name="Photocopy Shop", location="Library Block")

        create_counter(db, queue_id=q1.id, name="Counter 1")
        create_counter(db, queue_id=q1.id, name="Counter 2")
        create_counter(db, queue_id=q2.id, name="Counter 1")
        create_counter(db, queue_id=q3.id, name="Counter 1")
        print(f"  Done. Created 3 queues with counters.")
    else:
        print(f"  Done. Database already has {existing} queues.")
    db.close()

    print("\n  [2/4] Generating training data...")
    from ml.generate_demo_data import generate_demo_data
    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "demo_queue_data.csv")
    generate_demo_data(n_samples=1500, output_path=data_path)

    print("\n  [3/4] Training ML models...")
    from ml.train import train_and_evaluate_all
    models, results = train_and_evaluate_all(data_path)

    print("\n  Model Comparison:")
    print("  " + "-" * 56)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["MAE"]):
        print(f"  {name:25s} MAE={metrics['MAE']:7.1f}s  RMSE={metrics['RMSE']:7.1f}s  R2={metrics['R2']:.4f}")
    print("  " + "-" * 56)

    print("\n  [4/4] Seeding historical observations...")
    _seed_observations()

    print("\n" + "=" * 60)
    print("  Setup complete.")
    print("=" * 60)
    print("\n  Next step: python run.py server")
    print("  Then open http://localhost:8000 in your browser.")


def _seed_observations():
    import random
    from datetime import datetime, timedelta
    from backend.database.connection import SessionLocal
    from backend.database.models import QueueObservation, Queue

    db = SessionLocal()
    queues = db.query(Queue).all()
    if not queues:
        db.close()
        return

    base_time = datetime(2024, 1, 1, 8, 0, 0)
    count = 0

    for queue in queues:
        q_len = random.randint(0, 3)
        for i in range(200):
            ts = base_time + timedelta(minutes=i)
            arrival = random.uniform(0.5, 3.0)
            departure = random.uniform(0.3, 2.5)
            q_len = max(0, min(15, q_len + random.randint(-1, 1)))
            active = random.choice([1, 2])

            obs = QueueObservation(
                timestamp=ts,
                queue_id=queue.id,
                queue_length=q_len,
                arrival_rate=round(arrival, 2),
                departure_rate=round(departure, 2),
                active_counters=active,
            )
            db.add(obs)
            count += 1

    db.commit()
    db.close()
    print(f"  Done. Seeded {count} historical observations.")


def cmd_server():
    print("=" * 60)
    print("  Starting ML Queue Management Server")
    print("=" * 60)
    print("  Backend API:  http://localhost:8000")
    print("  API Docs:     http://localhost:8000/docs")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


def cmd_train():
    print("Generating fresh training data...")
    from ml.generate_demo_data import generate_demo_data
    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "demo_queue_data.csv")
    generate_demo_data(n_samples=1500, output_path=data_path)

    print("Training models...")
    from ml.train import train_and_evaluate_all
    models, results = train_and_evaluate_all(data_path)

    print("\nResults:")
    for name, m in sorted(results.items(), key=lambda x: x[1]["MAE"]):
        print(f"  {name:25s} MAE={m['MAE']:.1f}s  R2={m['R2']:.4f}")


def cmd_test():
    print("Running tests...")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
    )


def cmd_camera():
    camera_args = sys.argv[2:] if len(sys.argv) > 2 else []
    cmd = [sys.executable, "-m", "vision.camera_runner"] + camera_args
    print("Starting Camera Queue Monitor...")
    print("Make sure the backend server is running: python run.py server")
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def cmd_roi():
    camera_args = sys.argv[2:] if len(sys.argv) > 2 else []
    cmd = [sys.executable, "-m", "vision.camera_runner", "--setup"] + camera_args
    subprocess.run(cmd, cwd=PROJECT_ROOT)


COMMANDS = {
    "setup": cmd_setup,
    "server": cmd_server,
    "train": cmd_train,
    "test": cmd_test,
    "camera": cmd_camera,
    "roi": cmd_roi,
}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "server"

    if command in COMMANDS:
        COMMANDS[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
