import Dexie,{type Table} from "dexie";
export type QueueItem={id:string;entity:string;action:"create"|"update";payload:unknown;createdAt:number;retryCount:number;status:"pending"|"syncing"|"failed"};
type Meta={key:string;value:string};
class GyaanDB extends Dexie{queue!:Table<QueueItem,string>;meta!:Table<Meta,string>;constructor(){super("gyaan-saathi");this.version(2).stores({queue:"id,status,createdAt",meta:"key"}).upgrade(async tx=>{await tx.table("meta").put({key:"schema_version",value:"2"});});}}
export const db=new GyaanDB();
export async function getClientSessionId(){const existing=await db.meta.get("client_session_id");if(existing?.value)return existing.value;const id=crypto.randomUUID();await db.meta.put({key:"client_session_id",value:id});return id;}
export async function enqueue(item:Omit<QueueItem,"id"|"createdAt"|"retryCount"|"status">){const record={...item,id:crypto.randomUUID(),createdAt:Date.now(),retryCount:0,status:"pending" as const};await db.queue.add(record);return record;}
export async function queueStats(){const pending=await db.queue.where("status").equals("pending").count();const syncing=await db.queue.where("status").equals("syncing").count();const failed=await db.queue.where("status").equals("failed").count();return {pending,syncing,failed,total:pending+syncing+failed};}