import Dexie,{type Table} from "dexie";

export type QueueItem={id:string;entity:string;action:"create"|"update";payload:unknown;createdAt:number;retryCount:number;status:"pending"|"syncing"|"failed"};
export type DiagnosticSessionCredentials={sessionId:number;clientSessionId:string;sessionToken:string;expiresAt:string};
type Meta={key:string;value:string};

class GyaanDB extends Dexie{
  queue!:Table<QueueItem,string>;
  meta!:Table<Meta,string>;
  constructor(){
    super("gyaan-saathi");
    this.version(2).stores({queue:"id,status,createdAt",meta:"key"}).upgrade(async tx=>{
      await tx.table("meta").put({key:"schema_version",value:"2"});
    });
  }
}

export const db=new GyaanDB();

function uuid(){
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  throw new Error("secure_random_uuid_unavailable");
}

export async function getClientSessionId(){
  const existing=await db.meta.get("client_session_id");
  if(existing?.value)return existing.value;
  const id=uuid();
  await db.meta.put({key:"client_session_id",value:id});
  return id;
}

export async function saveDiagnosticSession(session:DiagnosticSessionCredentials){
  await db.meta.bulkPut([
    {key:"session_id",value:String(session.sessionId)},
    {key:"client_session_id",value:session.clientSessionId},
    {key:"session_token",value:session.sessionToken},
    {key:"session_expires_at",value:session.expiresAt},
  ]);
}

export async function getDiagnosticSession(){
  const values=await db.meta.bulkGet(["session_id","client_session_id","session_token","session_expires_at"]);
  if(values.some((v)=>!v?.value)) return null;
  return {
    sessionId:Number(values[0]!.value),
    clientSessionId:values[1]!.value,
    sessionToken:values[2]!.value,
    expiresAt:values[3]!.value,
  } satisfies DiagnosticSessionCredentials;
}

export async function clearDiagnosticSession(){
  await db.meta.bulkDelete(["session_id","client_session_id","session_token","session_expires_at"]);
}

export async function enqueue(item:Omit<QueueItem,"id"|"createdAt"|"retryCount"|"status">){
  const record={...item,id:uuid(),createdAt:Date.now(),retryCount:0,status:"pending" as const};
  await db.queue.add(record);
  return record;
}

export async function queueStats(){
  const pending=await db.queue.where("status").equals("pending").count();
  const syncing=await db.queue.where("status").equals("syncing").count();
  const failed=await db.queue.where("status").equals("failed").count();
  return {pending,syncing,failed,total:pending+syncing+failed};
}
