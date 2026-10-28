import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { adminLogout, getCameraFrameUrl } from "../services/api";

const authHdr = () => {
  const t = sessionStorage.getItem("adminToken");
  return t ? { Authorization: `Bearer ${t}` } : {};
};

export default function AdminDashboard() {
  const [queues, setQueues] = useState([]);
  const [live, setLive] = useState({});
  const [history, setHistory] = useState([]);
  const [selQ, setSelQ] = useState(1);
  const [adj, setAdj] = useState({});
  const [camOn, setCamOn] = useState(false);
  const [camQid, setCamQid] = useState(null);
  const [camCnt, setCamCnt] = useState(0);
  const [camErr, setCamErr] = useState("");
  const [camFrame, setCamFrame] = useState(null);
  const [camIdx, setCamIdx] = useState(0);

  useEffect(() => {
    fetch("/api/queues").then(r=>r.json()).then(d => { setQueues(d); if(d.length) setSelQ(d[0].id); }).catch(()=>{});
  }, []);

  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const c = await fetch("/api/camera/status").then(r=>r.json());
        setCamOn(c.status==="running"); setCamCnt(c.queue_count||0);
        if(c.status==="running") { const f=await getCameraFrameUrl(); if(f.ok){const b=await f.blob();const u=URL.createObjectURL(b);setCamFrame(p=>{if(p)URL.revokeObjectURL(p);return u;});} }
        else setCamFrame(null);
        const qs = await fetch("/api/queues").then(r=>r.json());
        for(const q of qs){const l=await fetch(`/api/queues/${q.id}/live`).then(r=>r.json());setLive(p=>({...p,[q.id]:l}));}
        const h = await fetch(`/api/queues/${selQ}/history?limit=50`).then(r=>r.json());
        setHistory(h.reverse());
      } catch(e) {}
    }, 500);
    return () => clearInterval(iv);
  }, [selQ]);

  const adjust = async (id, d) => { setAdj(p=>({...p,[id]:true})); try{await fetch(`/api/queues/${id}/adjust?delta=${d}`,{method:"POST",headers:authHdr()});}catch(e){} setTimeout(()=>setAdj(p=>({...p,[id]:false})),300); };

  const camStart = async () => {
    setCamErr("");
    try {
      if(camOn){await fetch("/api/camera/stop",{method:"POST",headers:authHdr()});await new Promise(r=>setTimeout(r,1000));}
      const r=await fetch(`/api/camera/start?queue_id=${selQ}&camera=${camIdx}`,{method:"POST",headers:authHdr()});
      const d=await r.json();
      if(d.status==="error"){setCamErr(d.message||"Cannot open camera");return;}
      setCamQid(selQ);
    } catch(e){setCamErr("Error: "+e.message);}
  };

  const camStop = async () => { try{await fetch("/api/camera/stop",{method:"POST",headers:authHdr()});}catch(e){} setCamOn(false);setCamQid(null);setCamFrame(null);setCamErr(""); };

  const nav = useNavigate();
  const logout = async () => {
    try { await adminLogout(); } catch(e) {}
    sessionStorage.removeItem("adminToken");
    window.dispatchEvent(new Event("admin-auth-changed"));
    nav("/admin/login");
  };

  if(!queues.length) return <div style={{padding:40,textAlign:"center",color:"#94a3b8"}}>Loading...</div>;

  return (
    <div style={{maxWidth:1000,margin:"0 auto",padding:"20px 20px 40px",minHeight:"calc(100vh - 52px)"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16}}>
        <h1 style={{fontSize:22,fontWeight:700,letterSpacing:-0.3,color:"#0f172a"}}>Admin</h1>
        <button onClick={logout} style={{padding:"6px 14px",background:"#ef4444",color:"#fff",border:"none",borderRadius:6,fontSize:12,fontWeight:600,cursor:"pointer"}}>Logout</button>
      </div>

      {/* Camera */}
      <div style={{background:"#fff",borderRadius:10,padding:16,marginBottom:12,border:"1px solid #f1f5f9",boxShadow:"0 1px 2px rgba(0,0,0,0.04)"}}>
        <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
          <span style={{fontSize:14,fontWeight:700}}>Camera</span>
          {camOn && <span style={{fontSize:11,fontWeight:600,color:"#10b981",display:"flex",alignItems:"center",gap:4}}><span style={{width:6,height:6,borderRadius:"50%",background:"#10b981",animation:"pulse 1.5s infinite"}}/>ON</span>}
        </div>
        <div style={{display:"flex",gap:12}}>
          <div style={{flex:"0 0 220px",display:"flex",flexDirection:"column",gap:6}}>
            <select value={selQ} onChange={e=>setSelQ(Number(e.target.value))} style={{padding:"7px 10px",fontSize:13,borderRadius:6,border:"1px solid #e2e8f0"}}>
              {queues.map(q=><option key={q.id} value={q.id}>{q.name}</option>)}
            </select>
            <select value={camIdx} onChange={e=>setCamIdx(Number(e.target.value))} style={{padding:"7px 10px",fontSize:13,borderRadius:6,border:"1px solid #e2e8f0"}}>
              <option value={0}>Built-in Webcam</option><option value={1}>External Camera</option>
            </select>
            {!camOn ? (
              <button onClick={camStart} style={{padding:"8px",background:"#10b981",color:"#fff",border:"none",borderRadius:6,fontSize:13,fontWeight:600,cursor:"pointer"}}>Start Camera</button>
            ) : (
              <button onClick={camStop} style={{padding:"8px",background:"#ef4444",color:"#fff",border:"none",borderRadius:6,fontSize:13,fontWeight:600,cursor:"pointer"}}>Stop Camera</button>
            )}
            {camErr && <div style={{background:"#fee2e2",color:"#ef4444",padding:"5px 8px",borderRadius:4,fontSize:11}}>{camErr}</div>}
            {camOn && <div style={{fontSize:11,color:"#64748b"}}>Queue #{camQid} | {camCnt} people</div>}
          </div>
          <div style={{flex:1,background:"#0a0a0a",borderRadius:6,minHeight:160,display:"flex",alignItems:"center",justifyContent:"center",overflow:"hidden"}}>
            {camFrame ? <img src={camFrame} alt="" style={{width:"100%",maxHeight:240,objectFit:"contain"}} /> : <span style={{color:"#64748b",fontSize:12}}>{camOn?"Starting...":"Click Start"}</span>}
          </div>
        </div>
      </div>

      {/* Queues */}
      <div style={{background:"#fff",borderRadius:10,padding:16,marginBottom:12,border:"1px solid #f1f5f9",boxShadow:"0 1px 2px rgba(0,0,0,0.04)"}}>
        <span style={{fontSize:14,fontWeight:700,display:"block",marginBottom:10}}>Queues</span>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(180px, 1fr))",gap:8}}>
          {queues.map(q => {
            const l=live[q.id]||{}; const ql=l.queue_length??0; const w=l.estimated_wait_minutes??0; const st=l.status??"low";
            const sc=st==="low"?"#10b981":st==="moderate"?"#f59e0b":"#ef4444"; const bg=st==="low"?"#d1fae5":st==="moderate"?"#fef3c7":"#fee2e2";
            const isAdj=adj[q.id]; const isCam=q.id===camQid;
            return (
              <div key={q.id} style={{borderRadius:8,border:isCam?"2px solid #3b82f6":"1px solid #e2e8f0",overflow:"hidden",background:"#fff"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 10px",borderBottom:"1px solid #f1f5f9",background:"#f8fafc"}}>
                  <span style={{fontSize:11,fontWeight:600}}>{q.name}</span>
                  <div style={{display:"flex",gap:3}}>
                    {isCam && <span style={{background:"#3b82f6",color:"#fff",padding:"1px 4px",borderRadius:2,fontSize:8,fontWeight:700}}>CAM</span>}
                    <span style={{padding:"1px 6px",borderRadius:3,fontSize:9,fontWeight:600,background:bg,color:sc}}>{st}</span>
                  </div>
                </div>
                <div style={{padding:"10px 8px",textAlign:"center"}}>
                  <div style={{fontSize:28,fontWeight:800,lineHeight:1}}>{ql}</div>
                  <div style={{fontSize:11,color:"#3b82f6",fontWeight:600,marginTop:3}}>{w>0?`${w} min`:"--"}</div>
                </div>
                <div style={{display:"flex",justifyContent:"center",gap:3,padding:"5px 6px",borderTop:"1px solid #f1f5f9"}} onClick={e=>e.stopPropagation()}>
                  <button onClick={()=>adjust(q.id,-5)} disabled={isAdj||ql<=0} style={{border:"none",borderRadius:3,color:"#fff",fontWeight:700,fontSize:10,padding:"3px 7px",cursor:"pointer",background:"#dc2626"}}>-5</button>
                  <button onClick={()=>adjust(q.id,-1)} disabled={isAdj||ql<=0} style={{border:"none",borderRadius:3,color:"#fff",fontWeight:700,fontSize:10,padding:"3px 7px",cursor:"pointer",background:"#ef4444"}}>-1</button>
                  <span style={{fontSize:13,fontWeight:800,minWidth:24,textAlign:"center"}}>{ql}</span>
                  <button onClick={()=>adjust(q.id,1)} disabled={isAdj} style={{border:"none",borderRadius:3,color:"#fff",fontWeight:700,fontSize:10,padding:"3px 7px",cursor:"pointer",background:"#10b981"}}>+1</button>
                  <button onClick={()=>adjust(q.id,5)} disabled={isAdj} style={{border:"none",borderRadius:3,color:"#fff",fontWeight:700,fontSize:10,padding:"3px 7px",cursor:"pointer",background:"#059669"}}>+5</button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Charts */}
      <div style={{background:"#fff",borderRadius:10,padding:16,border:"1px solid #f1f5f9",boxShadow:"0 1px 2px rgba(0,0,0,0.04)"}}>
        <span style={{fontSize:14,fontWeight:700,display:"block",marginBottom:10}}>History</span>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          <div style={{background:"#f8fafc",padding:12,borderRadius:6}}>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="timestamp" tick={{fontSize:8}} stroke="#94a3b8"/>
                <YAxis stroke="#94a3b8" fontSize={10}/>
                <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:6,fontSize:11}}/>
                <Line type="monotone" dataKey="queue_length" stroke="#3b82f6" strokeWidth={2} dot={false}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{background:"#f8fafc",padding:12,borderRadius:6}}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={history.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="timestamp" tick={{fontSize:8}} stroke="#94a3b8"/>
                <YAxis stroke="#94a3b8" fontSize={10}/>
                <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:6,fontSize:11}}/>
                <Legend/>
                <Bar dataKey="arrival_rate" fill="#10b981" name="In" radius={[2,2,0,0]}/>
                <Bar dataKey="departure_rate" fill="#ef4444" name="Out" radius={[2,2,0,0]}/>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
