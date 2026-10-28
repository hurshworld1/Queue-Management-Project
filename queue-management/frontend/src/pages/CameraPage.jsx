import React, { useState, useEffect, useCallback } from "react";
import { getQueues, startCamera, stopCamera, getCameraStatus, getCameraFrameUrl } from "../services/api";

export default function CameraPage() {
  const [queues, setQueues] = useState([]);
  const [selQ, setSelQ] = useState(1);
  const [status, setStatus] = useState("stopped");
  const [stats, setStats] = useState({});
  const [frame, setFrame] = useState(null);
  const [error, setError] = useState(null);
  const [camIdx, setCamIdx] = useState(0);

  useEffect(() => {
    getQueues().then(r => { setQueues(r.data); if(r.data.length) setSelQ(r.data[0].id); }).catch(()=>{});
  }, []);

  const poll = useCallback(async () => {
    try {
      const r = await getCameraStatus();
      setStats(r.data);
      setStatus(r.data.status || "stopped");
      if(r.data.status === "running") {
        const f = await getCameraFrameUrl();
        if(f.ok) { const b = await f.blob(); const u = URL.createObjectURL(b); setFrame(p => { if(p) URL.revokeObjectURL(p); return u; }); }
      } else { setFrame(null); }
    } catch(e) {}
  }, []);

  useEffect(() => { const iv = setInterval(poll, 300); return () => clearInterval(iv); }, [poll]);

  const start = async () => {
    setError(null);
    try {
      const r = await startCamera(selQ, camIdx);
      if(r.data.status === "error") { setError(r.data.message || "Cannot open camera"); return; }
      setStatus("starting");
      const p = setInterval(async () => {
        const r = await getCameraStatus();
        setStats(r.data);
        if(r.data.status === "running") { clearInterval(p); setStatus("running"); }
        else if(r.data.status?.startsWith("error")) { clearInterval(p); setError(r.data.status); setStatus("stopped"); }
      }, 500);
      setTimeout(() => clearInterval(p), 10000);
    } catch(e) { setError("Failed: " + e.message); }
  };

  const stop = async () => { await stopCamera(); setStatus("stopped"); setFrame(null); setStats({}); };

  const running = status === "running" || status === "starting";

  return (
    <div style={{maxWidth:800,margin:"0 auto",padding:"20px 20px 40px",minHeight:"calc(100vh - 52px)"}}>
      {/* Controls */}
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14,flexWrap:"wrap"}}>
        <select value={selQ} onChange={e=>setSelQ(Number(e.target.value))} style={{padding:"7px 10px",fontSize:13,borderRadius:6,border:"1px solid #e2e8f0"}} disabled={running}>
          {queues.map(q=><option key={q.id} value={q.id}>{q.name}</option>)}
        </select>
        <select value={camIdx} onChange={e=>setCamIdx(Number(e.target.value))} style={{padding:"7px 10px",fontSize:13,borderRadius:6,border:"1px solid #e2e8f0"}} disabled={running}>
          <option value={0}>Webcam</option><option value={1}>External</option>
        </select>
        {!running ? (
          <button onClick={start} style={{padding:"7px 20px",background:"#10b981",color:"#fff",border:"none",borderRadius:6,fontSize:13,fontWeight:600,cursor:"pointer"}}>Start</button>
        ) : (
          <button onClick={stop} style={{padding:"7px 20px",background:"#ef4444",color:"#fff",border:"none",borderRadius:6,fontSize:13,fontWeight:600,cursor:"pointer"}}>Stop</button>
        )}
        {running && <span style={{fontSize:12,color:"#64748b",marginLeft:8}}>{stats.queue_count ?? 0} people detected</span>}
      </div>
      {error && <div style={{background:"#fee2e2",color:"#ef4444",padding:"6px 12px",borderRadius:6,marginBottom:12,fontSize:12}}>{error}</div>}

      {/* Video */}
      <div style={{background:"#0a0a0a",borderRadius:10,overflow:"hidden",minHeight:340,display:"flex",alignItems:"center",justifyContent:"center",position:"relative",boxShadow:"0 2px 8px rgba(0,0,0,0.12)"}}>
        {frame ? (
          <img src={frame} alt="" style={{width:"100%",maxHeight:440,objectFit:"contain"}} />
        ) : (
          <div style={{textAlign:"center",padding:48}}>
            {running ? (
              <div style={{width:32,height:32,border:"3px solid #333",borderTop:"3px solid #10b981",borderRadius:"50%",animation:"spin 1s linear infinite",margin:"0 auto"}}/>
            ) : (
              <div style={{color:"#64748b",fontSize:14}}>Click Start to begin</div>
            )}
          </div>
        )}
        {running && frame && (
          <div style={{position:"absolute",top:8,left:8,background:"rgba(239,68,68,0.9)",color:"#fff",padding:"3px 8px",borderRadius:4,fontSize:10,fontWeight:700,display:"flex",alignItems:"center",gap:5}}>
            <span style={{width:5,height:5,borderRadius:"50%",background:"#fff",animation:"pulse 1.5s infinite"}}/>LIVE
          </div>
        )}
      </div>
    </div>
  );
}
