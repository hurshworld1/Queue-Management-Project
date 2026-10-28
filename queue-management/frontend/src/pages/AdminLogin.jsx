import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { adminLogin } from "../services/api";

export default function AdminLogin() {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await adminLogin(user, pass);
      sessionStorage.setItem("adminToken", res.data.token);
      window.dispatchEvent(new Event("admin-auth-changed"));
      nav("/admin");
    } catch (err) {
      if (err.response && err.response.status === 401) {
        setError("Invalid username or password");
      } else {
        setError("Cannot reach server. Try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{minHeight:"calc(100vh - 52px)",background:"#f8fafc",display:"flex",alignItems:"center",justifyContent:"center"}}>
      <div style={{width:360,background:"#fff",borderRadius:12,padding:32,boxShadow:"0 4px 12px rgba(0,0,0,0.08)",border:"1px solid #f1f5f9"}}>
        <h2 style={{fontSize:20,fontWeight:700,color:"#0f172a",marginBottom:4,textAlign:"center"}}>Admin Login</h2>
        <p style={{fontSize:13,color:"#64748b",marginBottom:24,textAlign:"center"}}>Enter credentials to access admin panel</p>

        <form onSubmit={handleLogin}>
          <div style={{marginBottom:14}}>
            <label style={{display:"block",fontSize:12,fontWeight:600,color:"#64748b",marginBottom:4}}>Username</label>
            <input
              type="text"
              value={user}
              onChange={e => setUser(e.target.value)}
              placeholder="Enter username"
              style={{width:"100%",padding:"10px 12px",fontSize:14,borderRadius:8,border:"1px solid #e2e8f0",outline:"none",boxSizing:"border-box"}}
              required
            />
          </div>

          <div style={{marginBottom:14}}>
            <label style={{display:"block",fontSize:12,fontWeight:600,color:"#64748b",marginBottom:4}}>Password</label>
            <input
              type="password"
              value={pass}
              onChange={e => setPass(e.target.value)}
              placeholder="Enter password"
              style={{width:"100%",padding:"10px 12px",fontSize:14,borderRadius:8,border:"1px solid #e2e8f0",outline:"none",boxSizing:"border-box"}}
              required
            />
          </div>

          {error && (
            <div style={{background:"#fee2e2",color:"#ef4444",padding:"8px 12px",borderRadius:6,fontSize:12,marginBottom:14}}>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{width:"100%",padding:"10px",background:"#3b82f6",color:"#fff",border:"none",borderRadius:8,fontSize:14,fontWeight:600,cursor:"pointer",opacity:loading?0.7:1}}
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <div style={{marginTop:16,textAlign:"center",fontSize:11,color:"#94a3b8"}}>
          Contact your administrator for credentials
        </div>
      </div>
    </div>
  );
}
