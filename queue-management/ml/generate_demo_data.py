import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_demo_data(
    n_samples: int = 1000,
    n_counters: int = 3,
    output_path: str = None,
) -> pd.DataFrame:
    """Generate realistic synthetic queue observation data for model training."""
    np.random.seed(42)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "data", "demo_queue_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = []
    base_time = datetime(2024, 1, 1, 8, 0, 0)

    for i in range(n_samples):
        timestamp = base_time + timedelta(minutes=i)

        # Time-based patterns
        hour = timestamp.hour
        is_peak = hour in (12, 13, 17, 18)  # lunch and evening peaks
        is_lunch = 11 <= hour <= 14

        # Random queue dynamics
        active_counters = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        base_service_rate = 0.5 + active_counters * 0.3  # people/min

        if is_peak:
            arrival_rate = np.random.uniform(2.0, 4.0)
            base_service_rate *= 0.7  # slower during peaks
        else:
            arrival_rate = np.random.uniform(0.5, 2.0)

        departure_rate = min(arrival_rate * np.random.uniform(0.7, 1.1), base_service_rate)

        # Queue length evolves
        if i == 0:
            queue_length = np.random.randint(0, 5)
        else:
            change = arrival_rate - departure_rate
            queue_length = max(0, int(queue_length + change + np.random.normal(0, 0.5)))

        # Service time varies
        base_service_time = np.random.uniform(20, 60)  # seconds
        if is_lunch:
            base_service_time *= 1.3
        mean_service_time = base_service_time + np.random.normal(0, 5)
        median_service_time = mean_service_time + np.random.normal(0, 2)
        service_time_variability = abs(np.random.normal(5, 3))

        # Queue growth rate
        queue_growth_rate = (arrival_rate - departure_rate) * 60  # per minute

        # Recent waiting history
        recent_waiting_history = max(0, queue_length * mean_service_time / active_counters)

        # Time features
        time_of_day = hour + timestamp.minute / 60.0
        day_of_week = timestamp.weekday()

        # Actual waiting time (target)
        if queue_length > 0:
            actual_wait = (queue_length / max(departure_rate, 0.1)) * 60  # seconds
            actual_wait += np.random.normal(0, actual_wait * 0.1)  # noise
            actual_wait = max(0, actual_wait)
        else:
            actual_wait = 0

        # Predicted wait (what baseline would predict)
        predicted_wait = (queue_length / max(departure_rate, 0.1)) * 60

        # Anomaly flag
        recent_anomaly_flag = 1 if (service_time_variability > 10 and queue_length > 8) else 0

        data.append({
            "timestamp": timestamp.isoformat(),
            "queue_id": 1,
            "counter_id": np.random.randint(1, active_counters + 1),
            "queue_length": queue_length,
            "arrival_rate": round(arrival_rate, 2),
            "departure_rate": round(departure_rate, 2),
            "active_counters": active_counters,
            "mean_service_time": round(mean_service_time, 1),
            "median_service_time": round(median_service_time, 1),
            "service_time_variability": round(service_time_variability, 2),
            "queue_growth_rate": round(queue_growth_rate, 2),
            "recent_waiting_history": round(recent_waiting_history, 1),
            "time_of_day": round(time_of_day, 2),
            "day_of_week": day_of_week,
            "recent_anomaly_flag": recent_anomaly_flag,
            "estimated_service_rate": round(base_service_rate, 2),
            "predicted_wait": round(predicted_wait, 1),
            "actual_wait": round(actual_wait, 1),
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} samples → {output_path}")
    return df


if __name__ == "__main__":
    generate_demo_data()
