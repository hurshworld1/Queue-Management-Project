import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

export default function App() {
  const [isAuth, setIsAuth] = useState(!!sessionStorage.getItem("adminToken"));
  useEffect(() => {
    const sync = () => setIsAuth(!!sessionStorage.getItem("adminToken"));
    window.addEventListener("admin-auth-changed", sync);
    window.addEventListener("storage", sync);
    return () => { window.removeEventListener("admin-auth-changed", sync); window.removeEventListener("storage", sync); };
  }, []);

  return (
    <div style={{minHeight:"calc(100vh - 52px)",background:"#f8fafc",display:"flex",alignItems:"center",justifyContent:"center"}}>
      <div style={{textAlign:"center",padding:"0 32px",maxWidth:600}}>
        <h1 style={{fontSize:48,fontWeight:800,letterSpacing:-2,color:"#0f172a",lineHeight:1.1,marginBottom:12}}>
          Queue<span style={{color:"#3b82f6"}}>Vision</span>
        </h1>
        <p style={{fontSize:16,color:"#64748b",marginBottom:40,lineHeight:1.6}}>
          Real-time queue monitoring with AI-powered waiting time estimation
        </p>
        <div style={{display:"flex",gap:12,justifyContent:"center"}}>
          <Link to="/customer" style={btnBlue}>View Queues</Link>
          {isAuth ? (
            <Link to="/admin" style={btnWhite}>Admin Panel</Link>
          ) : (
            <Link to="/admin/login" style={btnWhite}>Admin Login</Link>
          )}
        </div>
      </div>
    </div>
  );
}

const btnBlue = {
  padding: "12px 32px",
  background: "#3b82f6",
  color: "#fff",
  borderRadius: 10,
  textDecoration: "none",
  fontSize: 15,
  fontWeight: 600,
  boxShadow: "0 2px 8px rgba(59,130,246,0.3)"
};

const btnWhite = {
  padding: "12px 32px",
  background: "#fff",
  color: "#0f172a",
  borderRadius: 10,
  border: "1px solid #e2e8f0",
  textDecoration: "none",
  fontSize: 15,
  fontWeight: 500
};
