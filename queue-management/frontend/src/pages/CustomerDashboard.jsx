import React, { useState, useEffect, useCallback } from "react";
import { getQueues, getLiveStatus, connectLiveWS } from "../services/api";

export default function CustomerDashboard() {
  const [queues, setQueues] = useState([]);
  const [liveData, setLiveData] = useState({});
  const [loading, setLoading] = useState(true);
  const [camRunning, setCamRunning] = useState(false);

  const loadQueues = useCallback(async () => {
    try {
      const res = await getQueues();
      setQueues(res.data);
      for (const q of res.data) {
        try { const l = await getLiveStatus(q.id); setLiveData(p => ({...p, [q.id]: l.data})); } catch(e) {}
      }
    } catch(e) {} finally { setLoading(false); }
  }, []);

  useEffect(() => {
    loadQueues();
    const ws = connectLiveWS(s => s.forEach(x => setLiveData(p => ({...p, [x.queue_id]: x}))));
    return () => ws.close();
  }, []);

  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const cr = await fetch("/api/camera/status");
        if (cr.ok) {
          const c = await cr.json();
          setCamRunning(c.status === "running");
          if (c.status === "running" && c.queue_count !== undefined) {
            setLiveData(p => {
              const id = queues.length > 0 ? queues[0].id : 1;
              const e = p[id] || {};
              return {...p, [id]: {...e, queue_id:id, queue_length:c.queue_count}};
            });
          }
        }
        for (const q of queues) {
          try { const l = await getLiveStatus(q.id); setLiveData(p => ({...p, [q.id]: l.data})); } catch(e) {}
        }
      } catch(e) {}
    }, 500);
    return () => clearInterval(poll);
  }, [queues]);

  if (loading) return <div style={{display:"flex",justifyContent:"center",alignItems:"center",height:"60vh",color:"#94a3b8"}}>Loading...</div>;

  return (
    <div style={{maxWidth:800,margin:"0 auto",padding:"20px 20px 40px",minHeight:"calc(100vh - 52px)"}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}>
        <h1 style={{fontSize:22,fontWeight:700,letterSpacing:-0.3,color:"#0f172a",margin:0}}>Queue Status</h1>
        {camRunning && (
          <span style={{background:"rgba(239,68,68,0.9)",color:"#fff",padding:"2px 8px",borderRadius:4,fontSize:10,fontWeight:700,display:"flex",alignItems:"center",gap:5}}>
            <span style={{width:5,height:5,borderRadius:"50%",background:"#fff",animation:"pulse 1.5s infinite"}}/>LIVE
          </span>
        )}
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(240px, 1fr))",gap:12}}>
        {queues.map((q, i) => {
          const l = liveData[q.id] || {};
          const ql = l.queue_length ?? 0;
          const wait = l.estimated_wait_minutes ?? 0;
          const st = l.status ?? "low";
          const avg = l.avg_service_time;
          const sc = st==="low"?"#10b981":st==="moderate"?"#f59e0b":"#ef4444";
          const bg = st==="low"?"#d1fae5":st==="moderate"?"#fef3c7":"#fee2e2";
          return (
            <div key={q.id} style={{background:"#fff",borderRadius:10,border:"1px solid #f1f5f9",boxShadow:"0 1px 2px rgba(0,0,0,0.06)",animation:"slideUp 0.3s ease-out both",animationDelay:`${i*50}ms`}}>
              <div style={{padding:"12px 16px",borderBottom:"1px solid #f1f5f9",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <span style={{fontWeight:600,fontSize:13,color:"#0f172a"}}>{q.name}</span>
                <span style={{padding:"2px 8px",borderRadius:4,fontSize:10,fontWeight:600,background:bg,color:sc,textTransform:"uppercase"}}>{st}</span>
              </div>
              <div style={{padding:"20px 16px",textAlign:"center"}}>
                <div style={{fontSize:44,fontWeight:800,lineHeight:1,color:"#0f172a"}}>{ql}</div>
                <div style={{fontSize:12,color:"#94a3b8",marginTop:4}}>people waiting</div>
              </div>
              <div style={{display:"flex",borderTop:"1px solid #f1f5f9",background:"#f8fafc",borderRadius:"0 0 10px 10px"}}>
                <div style={{flex:1,textAlign:"center",padding:"12px 6px"}}>
                  <div style={{fontSize:9,color:"#94a3b8",textTransform:"uppercase",letterSpacing:0.5}}>Wait</div>
                  <div style={{fontSize:16,fontWeight:700,color:"#3b82f6",marginTop:1}}>{wait > 0 ? `${wait}m` : "--"}</div>
                </div>
                <div style={{width:1,background:"#e2e8f0"}}/>
                <div style={{flex:1,textAlign:"center",padding:"12px 6px"}}>
                  <div style={{fontSize:9,color:"#94a3b8",textTransform:"uppercase",letterSpacing:0.5}}>Avg</div>
                  <div style={{fontSize:14,fontWeight:600,color:"#0f172a",marginTop:1}}>{avg ? `${avg}s` : "--"}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {queues.length === 0 && (
        <div style={{textAlign:"center",padding:48,background:"#fff",borderRadius:10,color:"#64748b",border:"1px dashed #e2e8f0",fontSize:14}}>
          No queues configured
        </div>
      )}
    </div>
  );
}
