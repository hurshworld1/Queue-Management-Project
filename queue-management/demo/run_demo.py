"""
Demo Script: Set up the queue management system with sample data and train a model.

Usage:
    python -m demo.run_demo
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.connection import init_db, SessionLocal
from backend.database.models import Queue, Counter
from backend.services.queue_service import create_queue, create_counter
from ml.generate_demo_data import generate_demo_data
from ml.train import train_and_evaluate_all


def setup_database():
    """Initialize database and create sample queues."""
    print("📦 Initializing database...")
    init_db()
    db = SessionLocal()

    # Create sample queues
    existing = db.query(Queue).count()
    if existing == 0:
        print("Creating sample queues...")
        q1 = create_queue(db, name="Canteen Counter 1", location="Main Building")
        q2 = create_queue(db, name="Canteen Counter 2", location="Main Building")
        q3 = create_queue(db, name="Photocopy Shop", location="Library Block")

        create_counter(db, queue_id=q1.id, name="Counter 1")
        create_counter(db, queue_id=q1.id, name="Counter 2")
        create_counter(db, queue_id=q2.id, name="Counter 1")
        create_counter(db, queue_id=q3.id, name="Counter 1")

        print(f"  Created {db.query(Queue).count()} queues")
    else:
        print(f"  Database already has {existing} queues")

    db.close()


def generate_data():
    """Generate synthetic training data."""
    print("\n📊 Generating demo training data...")
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ml", "data", "demo_queue_data.csv"
    )
    generate_demo_data(n_samples=1000, output_path=data_path)
    return data_path


def train_model(data_path):
    """Train all ML models and evaluate them."""
    print("\n🤖 Training ML models...")
    models, results = train_and_evaluate_all(data_path)

    print("\n📈 Model Comparison:")
    print("-" * 60)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["MAE"]):
        print(f"  {name:25s} MAE={metrics['MAE']:7.1f}s  RMSE={metrics['RMSE']:7.1f}s  R²={metrics['R2']:.4f}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("  ML Queue Management System - Demo Setup")
    print("=" * 60)

    setup_database()
    data_path = generate_data()
    train_model(data_path)

    print("\n" + "=" * 60)
    print("  ✅ Setup complete!")
    print("=" * 60)
    print("\nTo run the application:")
    print("  Backend:  uvicorn backend.main:app --reload --port 8000")
    print("  Frontend: cd frontend && npm install && npm run dev")
    print("\nOpen http://localhost:5173 to view the dashboards.")


if __name__ == "__main__":
    main()
