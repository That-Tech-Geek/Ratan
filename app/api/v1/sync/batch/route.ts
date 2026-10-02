import {ensureSchema} from "../../../../../lib/server/db";
import {createHash} from "node:crypto";
export const runtime = "nodejs";
const MAX_ITEMS=20, MAX_ITEM_BYTES=32000;
const ALLOWED_ENTITIES=new Set(["diagnostic_response","likert_response"]);
const ALLOWED_ACTIONS=new Set(["create","update"]);
function stableHash(value:unknown){return createHash("sha256").update(JSON.stringify(value)).digest("hex");}
export async function POST(request:Request){
  try {
    const raw=await request.text();
    if(raw.length>MAX_ITEMS*MAX_ITEM_BYTES)return Response.json({error:"payload_too_large"},{status:413});
    const body=JSON.parse(raw) as {items?:unknown};
    if(!Array.isArray(body.items)||body.items.length>MAX_ITEMS)return Response.json({error:"batch_limit"},{status:400});
    await ensureSchema();
    const {default:postgres}=await import("postgres"); const sql=postgres(process.env.DATABASE_URL!,{max:5,prepare:false});
    const accepted:string[]=[],duplicate:string[]=[],rejected:Array<{id:string;reason:string}>=[];
    try { for(const rawItem of body.items){
      if(!rawItem||typeof rawItem!=="object"){rejected.push({id:"unknown",reason:"invalid_item"});continue;}
      const item=rawItem as Record<string,unknown>; const id=typeof item.id==="string"?item.id:"";
      const entity=typeof item.entity==="string"?item.entity:""; const action=typeof item.action==="string"?item.action:"";
      if(!/^[0-9a-f-]{36}$/i.test(id)){rejected.push({id:id||"unknown",reason:"invalid_id"});continue;}
      if(!ALLOWED_ENTITIES.has(entity)){rejected.push({id,reason:"invalid_entity"});continue;}
      if(!ALLOWED_ACTIONS.has(action)){rejected.push({id,reason:"invalid_action"});continue;}
      if(JSON.stringify(item).length>MAX_ITEM_BYTES){rejected.push({id,reason:"item_too_large"});continue;}
      const payloadHash=stableHash(item);
      const rows=await sql`INSERT INTO sync_events(idempotency_key,entity_type,action,payload_hash,payload,received_at) VALUES (${id},${entity},${action},${payloadHash},${JSON.stringify(item.payload??null)}::jsonb,NOW()) ON CONFLICT (idempotency_key) DO NOTHING RETURNING idempotency_key`;
      if(rows.length){await sql`INSERT INTO audit_logs(actor_type,actor_id,action,entity_type,entity_id) VALUES ("sync","browser",${action},${entity},${id})`;accepted.push(id);} else duplicate.push(id);
    }} finally {await sql.end({timeout:1});}
    return Response.json({accepted,duplicate,rejected,idempotency_ids:[...accepted,...duplicate],server_time:new Date().toISOString()});
  } catch { return Response.json({error:"sync_failed"},{status:500}); }
}