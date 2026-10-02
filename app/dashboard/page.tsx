"use client";
import {useEffect,useState} from "react";

import {subscribeAuth,currentIdToken,signOutUser} from "../../lib/firebase-client";
export default function Dashboard(){const[user,setUser]=useState<any>(null),[data,setData]=useState<any>(null),[error,setError]=useState("");
useEffect(()=>subscribeAuth(setUser),[]);
useEffect(()=>{if(user)void (async()=>{const t=await currentIdToken();if(!t)return;const r=await fetch("/api/v1/dashboard",{headers:{Authorization:"Bearer "+t}});if(r.ok)setData(await r.json());else setError("dashboard_unavailable");})();},[user]);
if(!user)return <main className="shell"><section className="card"><h1>Teacher dashboard</h1><p>Sign in from the main Gyaan Saathi page first.</p><a href="/">Go to sign in</a></section></main>;
if(!data)return <main className="shell"><section className="card"><p>Loading…</p>{error&&<p>{error}</p>}</section></main>;
return <main className="shell"><header className="topbar"><b>Teacher dashboard</b><div><a href="/">Start diagnostic</a> <button onClick={()=>void signOutUser()}>Sign out</button></div></header><section className="grid">{data.students.map((s:any)=><article className="card" key={s.id}><h3>{s.external_id}</h3><p>Class {s.class_no} · {s.medium}</p>{data.reports.filter((r:any)=>r.student_id===s.id).slice(0,3).map((r:any)=><div key={r.session_id}><b>{r.subject}</b>{r.topics.length?r.topics.map((t:any)=><p key={t.topic}>{t.weak?"🔴":"🟢"} {t.topic}: {t.accuracy}%</p>):<p className="muted">No responses yet.</p>}</div>)}</article>)}</section></main>;}