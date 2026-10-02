import {db} from "../../../../../lib/server/db";
import {requireTeacher} from "../../../../../lib/server/auth";
import {hashSessionToken} from "../../../../../lib/server/session-token";
import {rateLimit} from "../../../../../lib/server/rate-limit";
import {parseDiagnosticResponse} from "../../../../../lib/server/validation";
import {computePreference} from "../../../../../lib/server/likert-scoring";
import {createHash} from "node:crypto";
export const runtime="nodejs";
const MAX_ITEMS=20,MAX_ITEM_BYTES=32000,hash=(v:unknown)=>createHash("sha256").update(JSON.stringify(v)).digest("hex");
export async function POST(request:Request){try{
 const teacher=await requireTeacher(request);rateLimit("sync:"+teacher.uid,100,60000);const raw=await request.text();if(raw.length>MAX_ITEMS*MAX_ITEM_BYTES)return Response.json({error:"payload_too_large"},{status:413});
 let body:any;try{body=JSON.parse(raw)}catch{return Response.json({error:"invalid_json"},{status:400})}
 const sessionId=Number(body?.session_id),token=typeof body?.session_token==="string"?body.session_token:"",items=body?.items;
 if(!Number.isInteger(sessionId)||sessionId<=0||!token||!Array.isArray(items)||items.length>MAX_ITEMS)return Response.json({error:"invalid_batch"},{status:400});
 const sessions=await db()`SELECT ds.id,ds.student_id,ds.expires_at,ds.session_token_hash,s.school_id,ds.question_ids FROM diagnostic_sessions ds JOIN students s ON s.id=ds.student_id JOIN teachers t ON t.school_id=s.school_id WHERE ds.id=${sessionId} AND t.id=${teacher.teacherId} AND t.active=TRUE LIMIT 1`;
 const session=sessions[0];if(!session)return Response.json({error:"forbidden"},{status:403});if(new Date(session.expires_at).getTime()<=Date.now())return Response.json({error:"session_expired"},{status:401});if(hashSessionToken(token)!==session.session_token_hash)return Response.json({error:"invalid_session_token"},{status:401});
 const qids=new Set(Array.isArray(session.question_ids)?session.question_ids:[]),accepted:string[]=[],duplicate:string[]=[],rejected:any[]=[],likertResponses:any[]=[];
 await db().begin(async tx=>{for(const item of items){
   if(JSON.stringify(item).length>MAX_ITEM_BYTES){rejected.push({id:"unknown",reason:"item_too_large"});continue}
   const x=item as any;
   if(x.entity==="diagnostic_response"){const p=parseDiagnosticResponse(item);if(!p.ok){rejected.push({id:x.id||"unknown",reason:p.reason});continue}if(!qids.has(p.questionId)){rejected.push({id:p.id,reason:"question_not_in_session"});continue}
     const event=await tx`INSERT INTO sync_events(idempotency_key,entity_type,action,payload_hash,payload,received_at) VALUES(${p.id},'diagnostic_response','create',${hash(item)},${JSON.stringify({...p.payload,session_id:sessionId})}::jsonb,NOW()) ON CONFLICT(idempotency_key) DO NOTHING RETURNING id`;
     if(!event.length){duplicate.push(p.id);continue}
     await tx`INSERT INTO diagnostic_responses(session_id,question_id,selected_option,response_time_ms,synced_at,sync_event_id) VALUES(${sessionId},${p.questionId},${p.selectedOption},${p.responseTimeMs},NOW(),${event[0].id}) ON CONFLICT(session_id,question_id) DO NOTHING`;
     await tx`INSERT INTO audit_logs(actor_type,actor_id,action,entity_type,entity_id,school_id) VALUES('teacher',${teacher.uid},'create','diagnostic_response',${p.id},${teacher.schoolId})`;accepted.push(p.id);continue;
   }
   if(x.entity==="likert_response"){const id=typeof x.id==="string"?x.id:"";const p=x.payload as any;if(!/^[0-9a-f-]{36}$/i.test(id)||!p||typeof p!=="object"){rejected.push({id:id||"unknown",reason:"invalid_likert"});continue}const ls=Number(p.likert_session_id),score=Number(p.score),itemId=typeof p.item_id==="string"?p.item_id:"";const valid=await tx`SELECT id FROM likert_sessions WHERE id=${ls} AND student_id=${session.student_id}`;if(!valid.length||!itemId||!Number.isInteger(score)||score<1||score>5){rejected.push({id,reason:"invalid_likert"});continue}const event=await tx`INSERT INTO sync_events(idempotency_key,entity_type,action,payload_hash,payload,received_at) VALUES(${id},'likert_response','create',${hash(item)},${JSON.stringify({...p,session_id:sessionId})}::jsonb,NOW()) ON CONFLICT(idempotency_key) DO NOTHING RETURNING id`;if(!event.length){duplicate.push(id);continue}await tx`INSERT INTO likert_responses(session_id,item_id,score,synced_at,response_event_id) VALUES(${ls},${itemId},${score},NOW(),${event[0].id}) ON CONFLICT(session_id,item_id) DO NOTHING`;likertResponses.push({itemId,score});accepted.push(id);continue;}
   rejected.push({id:x.id||"unknown",reason:"unsupported_entity"});
 }});
 if(likertResponses.length){const ls=Number((items.find((x:any)=>x.entity==="likert_response") as any)?.payload?.likert_session_id||0);if(ls){const rows=await db()`SELECT student_id,item_id,score FROM likert_responses WHERE session_id=${ls}`;const pref=computePreference(rows.map(r=>({itemId:r.item_id,score:Number(r.score)})));await db()`INSERT INTO learning_preferences(student_id,likert_session_id,visual_score,auditory_score,reading_writing_score,kinesthetic_score,top_preference,second_preference,is_mixed) VALUES(${session.student_id},${ls},${pref.visual_score},${pref.auditory_score},${pref.reading_writing_score},${pref.kinesthetic_score},${pref.top_preference},${pref.second_preference},${pref.is_mixed}) ON CONFLICT(likert_session_id) DO UPDATE SET visual_score=EXCLUDED.visual_score,auditory_score=EXCLUDED.auditory_score,reading_writing_score=EXCLUDED.reading_writing_score,kinesthetic_score=EXCLUDED.kinesthetic_score,top_preference=EXCLUDED.top_preference,second_preference=EXCLUDED.second_preference,is_mixed=EXCLUDED.is_mixed,computed_at=NOW()`;}
 return Response.json({accepted,duplicate,rejected,idempotency_ids:[...accepted,...duplicate],server_time:new Date().toISOString()});
}catch(e){if(e instanceof Response)throw e;return Response.json({error:"sync_failed"},{status:500});}}