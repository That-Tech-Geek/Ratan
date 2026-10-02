import {db} from "../../../../lib/server/db";
export const runtime="nodejs";
export async function GET(){try{const r=await db()`SELECT (SELECT COUNT(*) FROM sync_events WHERE entity_type='diagnostic_response')::int AS sync_count,(SELECT COUNT(*) FROM diagnostic_responses)::int AS response_count`;return Response.json({ok:true,...r[0]});}catch{return Response.json({ok:false},{status:503});}}