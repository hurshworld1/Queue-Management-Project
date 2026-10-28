import React, { useState } from "react";
import { adjustQueueLength } from "../services/api";

const statusColors = {
  low: "#4CAF50",
  moderate: "#FF9800",
  high: "#F44336",
};

const trendIcons = {
  growing: "📈",
  shrinking: "📉",
  stable: "➡️",
};

export default function QueueCard({ queue, live, onQueueUpdate }) {
  const [adjusting, setAdjusting] = useState(false);
  const queueLength = live?.queue_length ?? 0;
  const estimatedWait = live?.estimated_wait_minutes ?? 0;
  const trend = live?.trend ?? "stable";
  const status = live?.status ?? "low";
  const lowerBound = live?.lower_bound_minutes;
  const upperBound = live?.upper_bound_minutes;

  const handleAdjust = async (delta) => {
    if (adjusting) return;
    setAdjusting(true);
    try {
      await adjustQueueLength(queue.id, delta);
      if (onQueueUpdate) onQueueUpdate();
    } catch (err) {
      console.error("Failed to adjust queue:", err);
    } finally {
      setAdjusting(false);
    }
  };

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <h3 style={styles.name}>{queue.name}</h3>
        <span
          style={{
            ...styles.statusBadge,
            background: statusColors[status],
          }}
        >
          {status.toUpperCase()}
        </span>
      </div>

      <div style={styles.stats}>
        <div style={styles.stat}>
          <span style={styles.statLabel}>People</span>
          <span style={styles.statValue}>{queueLength}</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Est. Wait</span>
          <span style={styles.statValue}>
            {estimatedWait > 0 ? `${estimatedWait} min` : "—"}
          </span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Trend</span>
          <span style={styles.statValue}>{trendIcons[trend] ?? "—"}</span>
        </div>
      </div>

      {/* Manual +/- buttons */}
      <div style={styles.controls}>
        <button
          style={{ ...styles.btn, ...styles.btnMinus }}
          onClick={() => handleAdjust(-1)}
          disabled={adjusting || queueLength <= 0}
          title="Remove 1 person from queue"
        >
          −1
        </button>
        <button
          style={{ ...styles.btn, ...styles.btnMinus5 }}
          onClick={() => handleAdjust(-5)}
          disabled={adjusting || queueLength <= 0}
          title="Remove 5 people from queue"
        >
          −5
        </button>
        <div style={styles.queueCount}>{queueLength}</div>
        <button
          style={{ ...styles.btn, ...styles.btnPlus5 }}
          onClick={() => handleAdjust(5)}
          disabled={adjusting}
          title="Add 5 people to queue"
        >
          +5
        </button>
        <button
          style={{ ...styles.btn, ...styles.btnPlus }}
          onClick={() => handleAdjust(1)}
          disabled={adjusting}
          title="Add 1 person to queue"
        >
          +1
        </button>
      </div>

      {lowerBound != null && upperBound != null && (
        <div style={styles.confidence}>
          Likely {lowerBound}–{upperBound} min
        </div>
      )}

      {live?.active_counters != null && (
        <div style={styles.counters}>
          {live.active_counters} counter{live.active_counters !== 1 ? "s" : ""} active
        </div>
      )}

      {live?.last_updated && (
        <div style={styles.updated}>
          Updated {new Date(live.last_updated).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    background: "white",
    borderRadius: 12,
    padding: 20,
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  name: {
    fontSize: 18,
    fontWeight: 600,
  },
  statusBadge: {
    color: "white",
    padding: "4px 10px",
    borderRadius: 6,
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 0.5,
  },
  stats: {
    display: "flex",
    justifyContent: "space-around",
    marginBottom: 12,
  },
  stat: {
    textAlign: "center",
  },
  statLabel: {
    display: "block",
    fontSize: 12,
    color: "#999",
    marginBottom: 4,
  },
  statValue: {
    display: "block",
    fontSize: 24,
    fontWeight: 700,
  },
  controls: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    marginBottom: 12,
    padding: "8px 0",
  },
  btn: {
    border: "none",
    borderRadius: 8,
    fontSize: 16,
    fontWeight: 700,
    cursor: "pointer",
    padding: "8px 14px",
    color: "white",
    transition: "opacity 0.15s",
    minWidth: 48,
  },
  btnPlus: {
    background: "#4CAF50",
  },
  btnPlus5: {
    background: "#2E7D32",
  },
  btnMinus: {
    background: "#F44336",
  },
  btnMinus5: {
    background: "#C62828",
  },
  queueCount: {
    fontSize: 28,
    fontWeight: 800,
    minWidth: 50,
    textAlign: "center",
    color: "#333",
  },
  confidence: {
    textAlign: "center",
    fontSize: 13,
    color: "#666",
    background: "#f0f0f0",
    padding: "6px 12px",
    borderRadius: 6,
    marginBottom: 8,
  },
  counters: {
    textAlign: "center",
    fontSize: 13,
    color: "#888",
  },
  updated: {
    textAlign: "center",
    fontSize: 11,
    color: "#bbb",
    marginTop: 8,
  },
};
