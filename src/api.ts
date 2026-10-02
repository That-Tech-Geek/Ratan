const API=import.meta.env.VITE_API_URL||"";
export async function getQuestions(classNo=8,subject="maths"){
 const r=await fetch(API+`/api/v1/diagnostic/questions?class=${classNo}&subject=${encodeURIComponent(subject)}`);
 if(!r.ok)throw new Error("question fetch failed");return r.json();
}
export async function syncBatch(items:unknown[]){
 const r=await fetch(API+"/api/v1/sync/batch/",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({items})});
 if(!r.ok)throw new Error("sync failed");return r.json();
}