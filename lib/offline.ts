import Dexie,{type Table} from "dexie";
export type QueueItem={id:string;entity:string;action:"create"|"update";payload:unknown;createdAt:number;retryCount:number;status:"pending"|"syncing"|"failed"};
class GyaanDB extends Dexie{queue!:Table<QueueItem,string>;constructor(){super("gyaan-saathi");this.version(1).stores({queue:"id,status,createdAt"});}}
export const db=new GyaanDB();
export async function enqueue(item:Omit<QueueItem,"id"|"createdAt"|"retryCount"|"status">){const record={...item,id:crypto.randomUUID(),createdAt:Date.now(),retryCount:0,status:"pending" as const};await db.queue.add(record);return record;}