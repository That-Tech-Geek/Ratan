import {db} from "../../../../../lib/server/db";
export const runtime="nodejs";
function authorized(request:Request){const secret=process.env.CRON_SECRET;return Boolean(secret)&&request.headers.get("authorization")===`Bearer ${secret}`;}
export async function GET(request:Request){if(!authorized(request))return new Response("Unauthorized",{status:401});try{
 const repaired=await db()`INSERT INTO diagnostic_responses(session_id,question_id,selected_option,response_time_ms,synced_at,sync_event_id)
 SELECT (payload->>'session_id')::bigint,payload->>'question_id',payload->>'selected_option',(payload->>'response_time_ms')::int,received_at,id
 FROM sync_events se LEFT JOIN diagnostic_responses dr ON dr.sync_event_id=se.id
 WHERE se.entity_type='diagnostic_response' AND dr.id IS NULL AND payload ? 'session_id'
 ON CONFLICT(session_id,question_id) DO NOTHING RETURNING id`;
 await db()`DELETE FROM diagnostic_responses WHERE synced_at < NOW()-INTERVAL '6 months'`;
 await db()`DELETE FROM likert_responses WHERE synced_at < NOW()-INTERVAL '6 months'`;
 await db()`DELETE FROM sync_events WHERE received_at < NOW()-INTERVAL '30 days'`;
 const queue=await db()`SELECT id,student_id FROM deletion_queue WHERE processed_at IS NULL ORDER BY queued_at LIMIT 100`;
 for(const item of queue){await db().begin(async tx=>{await tx`INSERT INTO audit_logs(actor_type,actor_id,action,entity_type,entity_id) VALUES('system','maintenance','delete','student',${String(item.student_id)})`;await tx`DELETE FROM learning_preferences WHERE student_id=${item.student_id}`;await tx`DELETE FROM likert_responses WHERE session_id IN (SELECT id FROM likert_sessions WHERE student_id=${item.student_id})`;await tx`DELETE FROM likert_sessions WHERE student_id=${item.student_id}`;await tx`DELETE FROM diagnostic_responses WHERE session_id IN (SELECT id FROM diagnostic_sessions WHERE student_id=${item.student_id})`;await tx`DELETE FROM diagnostic_sessions WHERE student_id=${item.student_id}`;await tx`DELETE FROM consents WHERE student_id=${item.student_id}`;await tx`UPDATE deletion_queue SET processed_at=NOW() WHERE id=${item.id}`;});}
 const counts=await db()`SELECT (SELECT COUNT(*) FROM sync_events WHERE entity_type='diagnostic_response')::int sync_count,(SELECT COUNT(*) FROM diagnostic_responses)::int response_count`;
 return Response.json({ok:true,repaired:repaired.length,deleted:queue.length,...counts});
}catch(e){return Response.json({ok:false,error:e instanceof Error?e.message:"maintenance_failed"},{status:500});}}