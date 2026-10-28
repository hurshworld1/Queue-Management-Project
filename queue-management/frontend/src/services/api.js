import axios from "axios";

const API_BASE = "/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

// Attach admin token to every request when present
api.interceptors.request.use((config) => {
  const token = sessionStorage.getItem("adminToken");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Auth
export const adminLogin = (username, password) =>
  api.post("/auth/login", { username, password });
export const adminLogout = () => api.post("/auth/logout");

// Queue endpoints
export const getQueues = () => api.get("/queues");
export const createQueue = (data) => api.post("/queues", data);
export const getQueue = (id) => api.get(`/queues/${id}`);

// Live status
export const getLiveStatus = (queueId) => api.get(`/queues/${queueId}/live`);

// Manual queue control (admin)
export const adjustQueueLength = (queueId, delta) =>
  api.post(`/queues/${queueId}/adjust`, null, { params: { delta } });
export const setQueueLength = (queueId, length) =>
  api.post(`/queues/${queueId}/set-length`, null, { params: { length } });

// History & Analytics
export const getQueueHistory = (queueId, limit = 100) =>
  api.get(`/queues/${queueId}/history`, { params: { limit } });
export const getQueueAnalytics = (queueId) => api.get(`/queues/${queueId}/analytics`);

// Alerts
export const getAlerts = (queueId) =>
  api.get("/alerts", { params: queueId ? { queue_id: queueId } : {} });

// Config
export const configureCounter = (data) => api.post("/config/counter", data);

// Model
export const getModelMetrics = () => api.get("/model/metrics");

// Health check
export const getHealth = () => api.get("/health");

// Camera control (admin)
export const startCamera = (queueId, cameraIndex) =>
  api.post(`/camera/start`, null, { params: { queue_id: queueId, camera: cameraIndex } });
export const stopCamera = () => api.post(`/camera/stop`);
export const getCameraStatus = () => api.get(`/camera/status`);
// Frame fetch with auth header (axios won't set headers on plain fetch)
export const getCameraFrameUrl = () => {
  const token = sessionStorage.getItem("adminToken");
  return fetch("/api/camera/frame", {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
};

// WebSocket for live updates
export const connectLiveWS = (onMessage) => {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const ws = new WebSocket(`${protocol}//${host}/ws/live`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error("WebSocket parse error:", e);
    }
  };

  ws.onerror = (error) => {
    console.warn("WebSocket not available, using polling fallback");
  };

  return ws;
};

export default api;
