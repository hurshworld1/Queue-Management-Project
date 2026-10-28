import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Link, useLocation, Navigate } from "react-router-dom";
import CustomerDashboard from "./pages/CustomerDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import CameraPage from "./pages/CameraPage";
import AdminLogin from "./pages/AdminLogin";
import App from "./App";

const isAuthed = () => !!sessionStorage.getItem("adminToken");
const AUTH_EVENT = "admin-auth-changed";

function useAuth() {
  const [authed, setAuthed] = useState(isAuthed());
  useEffect(() => {
    const sync = () => setAuthed(isAuthed());
    window.addEventListener(AUTH_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(AUTH_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return authed;
}

function Protected({ authed, children }) {
  return authed ? children : <Navigate to="/admin/login" replace />;
}

function NavBar({ authed }) {
  const loc = useLocation();
  const items = [
    { to: "/", label: "Home" },
    { to: "/customer", label: "Queues" },
    ...(authed ? [
      { to: "/camera", label: "Camera" },
      { to: "/admin", label: "Admin" },
    ] : []),
    ...(!authed ? [{ to: "/admin/login", label: "Admin Login" }] : []),
  ];
  return (
    <nav style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 24px",height:52,background:"#0f172a",position:"sticky",top:0,zIndex:100}}>
      <Link to="/" style={{display:"flex",alignItems:"center",gap:10,textDecoration:"none",color:"#fff",fontWeight:700,fontSize:16}}>
        <span style={{width:28,height:28,borderRadius:7,background:"linear-gradient(135deg,#3b82f6,#8b5cf6)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:800}}>Q</span>
        QueueVision
      </Link>
      <div style={{display:"flex",gap:2}}>
        {items.map(it => {
          const a = loc.pathname === it.to;
          return (
            <Link key={it.to} to={it.to} style={{color:a?"#fff":"rgba(255,255,255,0.5)",textDecoration:"none",padding:"7px 14px",fontSize:13,fontWeight:500,borderRadius:6,background:a?"rgba(255,255,255,0.1)":"transparent",transition:"all 0.15s"}}>
              {it.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}

function AppRoutes() {
  const authed = useAuth();
  return (
    <div>
      <NavBar authed={authed} />
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/customer" element={<CustomerDashboard />} />
        <Route path="/admin/login" element={authed ? <Navigate to="/admin" replace /> : <AdminLogin />} />
        <Route path="/camera" element={<Protected authed={authed}><CameraPage /></Protected>} />
        <Route path="/admin" element={<Protected authed={authed}><AdminDashboard /></Protected>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  </React.StrictMode>
);
