import {db} from "./db";import {syncBatch} from "./api";
export async function flushQueue(){
 const pending=await db.queue.where("status").equals("pending").limit(20).toArray();
 if(!pending.length)return {synced:0,failed:0};
 await Promise.all(pending.map(x=>db.queue.update(x.id,{status:"syncing"})));
 try{
  const result=await syncBatch(pending.map(x=>({id:x.id,entity:x.entity,action:x.action,payload:x.payload,created_at:x.createdAt})));
  await db.transaction("rw",db.queue,async()=>{for(const x of pending)await db.queue.delete(x.id);});
  return {synced:result.accepted||pending.length,failed:0};
 }catch(error){
  await Promise.all(pending.map(async x=>{const retry=x.retryCount+1;await db.queue.update(x.id,{status:retry>5?"failed":"pending",retryCount:retry});}));
  throw error;
 }
}
export function registerSync(){
 window.addEventListener("online",()=>{void flushQueue().catch(()=>undefined)});
 document.addEventListener("visibilitychange",()=>{if(document.visibilityState==="visible"&&navigator.onLine)void flushQueue().catch(()=>undefined)});
}